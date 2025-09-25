# --- 0. ИМПОРТЫ И НАСТРОЙКИ ---
from ollama import Client 
import pandas as pd
import json
import random
import os
import re
from datetime import datetime
from time import sleep
from faker import Faker

# --- 1. РАСШИРЕННАЯ И СТРУКТУРИРОВАННАЯ БАЗА ЗНАНИЙ ---

# Столбцы, данные для которых генерирует LLM (исходные данные)
BASE_COLUMNS = {
    'item_number': ['№', '№ п/п', '#', 'Поз.'],
    'article': ['Артикул', 'Код товара', 'SKU', 'Номенклатурный номер'],
    'vendor_code': ['Артикул производителя', 'Код поставщика', 'Vendor Code'],
    'product_name': ['Наименование товара', 'Описание', 'Номенклатура', 'Товар'],
    'product_group': ['Товарная группа', 'Категория', 'Раздел'],
    'manufacturer': ['Производитель', 'Бренд', 'Торговая марка'],
    'package_count': ['Кол-во упаковок', 'Кол-во мест', 'Число коробок'],
    'units_in_package': ['Шт. в упаковке', 'Ед. в месте', 'Фасовка'],
    'unit': ['Ед. изм.', 'Ед.', 'Шт.', 'Упак.'],
    'base_price': ['Цена по прайсу', 'Базовая цена', 'Цена до скидки'],
    'discount_percent': ['Скидка %', '% скидки', 'Дисконт, %'],
    'vat_rate': ['Ставка НДС', '% НДС', 'НДС'],
    'country_of_origin': ['Страна происхождения', 'Производство', 'Страна'],
    'customs_declaration_number': ['Номер ГТД', 'ГТД', 'НТД'],
    'notes': ['Примечание', 'Комментарий', 'Доп. информация']
}

# Столбцы, значения для которых рассчитываются кодом
CALCULATED_COLUMNS = {
    'quantity': ['Общее кол-во', 'Количество', 'Кол-во', 'Qty'],
    'price_after_discount': ['Цена со скидкой', 'Цена', 'Отпускная цена'],
    'total_sum_after_discount': ['Сумма без НДС', 'Сумма', 'Стоимость', 'Сумма со скидкой'],
    'vat_amount': ['Сумма НДС', 'НДС (сумма)', 'В т.ч. НДС'],
    'total_with_vat': ['Всего с НДС', 'Итого', 'К оплате', 'Сумма с НДС']
}

# Определяет, какие базовые столбцы НУЖНЫ для расчета каждого вычисляемого столбца
COLUMN_DEPENDENCIES = {
    'quantity': ['package_count', 'units_in_package'],
    'price_after_discount': ['base_price', 'discount_percent'],
    'total_sum_after_discount': ['base_price', 'discount_percent', 'package_count', 'units_in_package'],
    'vat_amount': ['base_price', 'discount_percent', 'package_count', 'units_in_package', 'vat_rate'],
    'total_with_vat': ['base_price', 'discount_percent', 'package_count', 'units_in_package', 'vat_rate']
}

# Полный словарь синонимов для всех столбцов
COLUMN_SYNONYMS = {**BASE_COLUMNS, **CALCULATED_COLUMNS}

# Поля для шапки документа
HEADER_FIELDS = {
    'invoice_number': ['Счет-фактура №', 'Счет на оплату №', 'Инвойс №', 'УПД №'],
    'invoice_date': ['от', 'Дата составления', 'Дата'],
    'supplier_name': ['Поставщик', 'Продавец', 'Исполнитель'],
    'supplier_details': ['ИНН/КПП Поставщика', 'Реквизиты Продавца'],
    'customer_name': ['Покупатель', 'Заказчик', 'Плательщик'],
    'customer_details': ['ИНН/КПП Покупателя', 'Реквизиты Покупателя'],
    'contract_info': ['Основание', 'Договор №', 'Контракт']
}

# Темы для генерации строк
ITEM_EXAMPLES = ["электроинструменты", "сантехника", "сетевое оборудование", "автозапчасти", "канцелярские товары", "строительные материалы", "офисная мебель", "программное обеспечение", "компьютерные комплектующие", "хозтовары"]

# Конфигурация диапазонов для генерации чисел.
# Это дает полный контроль над разнообразием и реалистичностью данных.
NUMBER_GENERATION_CONFIG = {
    'base_price':       {'type': 'uniform', 'min': 150.0, 'max': 75000.0, 'round': 2},
    'package_count':    {'type': 'int', 'min': 1, 'max': 50},
    'units_in_package': {'type': 'int', 'min': 1, 'max': 100},
    'discount_percent': {'type': 'choice', 'options': [0, 0, 0, 5, 10, 15, 20, 25], 'weights': [50, 10, 5, 10, 10, 5, 5, 5]}, # Скидки часто бывают нулевыми
    'vat_rate':         {'type': 'choice', 'options': [0, 10, 20], 'weights': [10, 20, 70]} # НДС 20% наиболее частый
}


# --- 2. МОДУЛЬНЫЕ ФУНКЦИИ ---

def extract_number(value, default=0.0):
    """Безопасно извлекает число из строки или числового типа."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r'[\d.,]+', value)
        if match:
            try:
                return float(match.group(0).replace(',', '.'))
            except (ValueError, TypeError):
                return default
    return default

def generate_header_data(client: Client, model_name: str, target_fields: dict, faker: Faker):
    """Генерирует данные для шапки документа, используя Faker для названий."""
    field_instructions = ", ".join([f'"{v}"' for v in target_fields.values()])
    fake_company_name = faker.company()
    
    prompt = f"""
    Твоя задача — сгенерировать ОДИН JSON-объект с реквизитами для шапки счета-фактуры.
    Правила:
    1. Ответ должен быть ОДНИМ валидным JSON-объектом.
    2. Объект должен содержать ТОЛЬКО ключи: {field_instructions}.
    3. Заполни объект правдоподобными, но вымышленными данными компании типа "{fake_company_name}".
    4. ВАЖНО: Не пиши никаких объяснений, верни только JSON.
    """
    try:
        response = client.chat(model=model_name, messages=[{'role': 'user', 'content': prompt}], format='json')
        data = json.loads(response['message']['content'].strip())
        return data if isinstance(data, dict) else None
    except Exception as e:
        print(f"Ошибка при генерации шапки: {e}")
        return None

def generate_base_row_data(client: Client, model_name: str, target_base_columns: dict, item_theme: str, number_config: dict):
    """
    Генерирует базовые данные для строки, используя числа, предварительно сгенерированные в Python.
    """
    # 1. Генерируем числа в Python на основе конфигурации
    pre_generated_numbers = {}
    for semantic_key, config in number_config.items():
        if semantic_key in target_base_columns:
            column_name = target_base_columns[semantic_key]
            value = None
            if config['type'] == 'uniform':
                value = round(random.uniform(config['min'], config['max']), config['round'])
            elif config['type'] == 'int':
                value = random.randint(config['min'], config['max'])
            elif config['type'] == 'choice':
                value = random.choices(config['options'], weights=config.get('weights'), k=1)[0]
            
            if value is not None:
                pre_generated_numbers[column_name] = value
    
    # 2. Создаем "инъекцию" для промпта
    values_injection = "\n".join([f'    "{k}": {v}' for k, v in pre_generated_numbers.items()])
    prompt_values_section = f"""
ДАННЫЕ ДЛЯ ОБЯЗАТЕЛЬНОГО ИСПОЛЬЗОВАНИЯ:
{{
{values_injection}
}}
"""
    
    # 3. Формируем финальный промпт
    column_instructions = ", ".join([f'"{v}"' for v in target_base_columns.values()])
    prompt = f"""
Твоя задача — сгенерировать ОДИН JSON-объект, представляющий одну строку из счета-фактуры на тему "{item_theme}".

ПРАВИЛА:
1. Ответ должен быть ОДНИМ валидным JSON-объектом.
2. Объект должен содержать ТОЛЬКО ключи: {column_instructions}.
3. Для полей, связанных с числами (цена, количество, скидка, НДС), ты ДОЛЖЕН ИСПОЛЬЗОВАТЬ ТОЧНЫЕ ЗНАЧЕНИЯ, которые я предоставлю ниже. НЕ придумывай свои числа.
4. Для остальных полей (наименование, артикул, производитель и т.д.) придумай реалистичные, но вымышленные данные, которые бы соответствовали указанным числам.
5. Для поля 'Ед. изм.' (или его синонима) используй ТОЛЬКО НАЗВАНИЕ ЕДИНИЦЫ (например: "шт.", "упак.", "кг"), БЕЗ ЦИФР.
6. НЕ пиши никаких объяснений, верни только JSON.

{prompt_values_section}
"""
    try:
        response = client.chat(model=model_name, messages=[{'role': 'user', 'content': prompt}], format='json')
        content = response['message']['content'].strip().replace('```json', '').replace('```', '')
        data = json.loads(content)
        return data if isinstance(data, dict) else None
    except Exception as e:
        print(f"Ошибка при генерации базовой строки: {e}")
        return None

def calculate_row_totals(row_data: dict, semantic_map: dict):
    """Рассчитывает вычисляемые поля на основе базовых данных строки."""
    try:
        package_count = extract_number(row_data.get(semantic_map.get('package_count')), default=1.0)
        units_in_package = extract_number(row_data.get(semantic_map.get('units_in_package')), default=1.0)
        base_price = extract_number(row_data.get(semantic_map.get('base_price')), default=0.0)
        discount_percent = extract_number(row_data.get(semantic_map.get('discount_percent')), default=0.0)
        vat_rate = extract_number(row_data.get(semantic_map.get('vat_rate')), default=0.0)

        quantity = package_count * units_in_package
        price_after_discount = base_price * (1 - discount_percent / 100)
        total_sum_after_discount = quantity * price_after_discount
        vat_amount = total_sum_after_discount * (vat_rate / 100)
        total_with_vat = total_sum_after_discount + vat_amount
        
        if 'quantity' in semantic_map:
            row_data[semantic_map['quantity']] = quantity
        if 'price_after_discount' in semantic_map:
            row_data[semantic_map['price_after_discount']] = round(price_after_discount, 2)
        if 'total_sum_after_discount' in semantic_map:
            row_data[semantic_map['total_sum_after_discount']] = round(total_sum_after_discount, 2)
        if 'vat_amount' in semantic_map:
            row_data[semantic_map['vat_amount']] = round(vat_amount, 2)
        if 'total_with_vat' in semantic_map:
            row_data[semantic_map['total_with_vat']] = round(total_with_vat, 2)
            
    except Exception as e:
        print(f"Критическая ошибка при расчете строки: {row_data}. Ошибка: {e}")
    return row_data

def generate_full_invoice(client: Client, model_name: str, faker: Faker):
    """Основная функция-оркестратор для сборки одного полного счета."""
    print("--- Генерация шапки документа ---")
    selected_header_keys = random.sample(list(HEADER_FIELDS.keys()), k=random.randint(5, len(HEADER_FIELDS)))
    target_header_fields = {key: random.choice(HEADER_FIELDS[key]) for key in selected_header_keys}
    header_data = generate_header_data(client, model_name, target_header_fields, faker)
    if not header_data:
        print("Не удалось сгенерировать шапку. Пропускаем.")
        return None, None
    
    num_rows = random.randint(10, 15)
    item_theme = random.choice(ITEM_EXAMPLES)
    print(f"--- Генерация таблицы из {num_rows} строк на тему '{item_theme}' ---")
    
    num_columns = random.randint(12, len(COLUMN_SYNONYMS) - 2)
    selected_keys = set(random.sample(list(COLUMN_SYNONYMS.keys()), k=num_columns))
    
    required_keys = set()
    for key in selected_keys:
        if key in COLUMN_DEPENDENCIES:
            required_keys.update(COLUMN_DEPENDENCIES[key])
            
    final_keys = sorted(list(selected_keys.union(required_keys)))
    target_table_columns_map = {key: random.choice(COLUMN_SYNONYMS[key]) for key in final_keys}
    
    base_keys_to_generate = [k for k in final_keys if k in BASE_COLUMNS]
    target_base_columns = {key: target_table_columns_map[key] for key in base_keys_to_generate}
    
    print(f"Итоговая структура столбцов: {list(target_table_columns_map.values())}")

    all_rows = []
    for i in range(num_rows):
        print(f"Генерация строки {i + 1}/{num_rows}...")
        base_row = generate_base_row_data(client, model_name, target_base_columns, item_theme, NUMBER_GENERATION_CONFIG)
        
        if base_row:
            full_row = calculate_row_totals(base_row, target_table_columns_map)
            all_rows.append(full_row)
        else:
            print(f"Пропуск строки {i+1} из-за ошибки генерации.")

    if not all_rows:
        print("Не удалось сгенерировать строки таблицы. Пропускаем.")
        return header_data, None
    
    final_columns_order = [target_table_columns_map[key] for key in final_keys if key in target_table_columns_map]
    df = pd.DataFrame(all_rows, columns=final_columns_order)
    return header_data, df

# --- 3. ОСНОВНОЙ СКРИПТ ---

if __name__ == "__main__":
    # --- КЛЮЧЕВЫЕ НАСТРОЙКИ ---
    
    # 1. Укажите IP-адрес и порт машины, где запущен сервер Ollama.
    OLLAMA_HOST = 'http://100.74.62.22:11434' 
    
    # 2. Укажите модель, которую вы хотите использовать
    OLLAMA_MODEL = 'gemma3:12b' 
    
    # 3. Остальные настройки
    NUM_INVOICES_TO_GENERATE = 20
    OUTPUT_DIR = "generated_invoices"

    # --- ЗАПУСК ГЕНЕРАТОРА ---
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    faker = Faker('ru_RU')
    
    print(f"Генератор запущен.")
    print(f"Подключение к серверу Ollama по адресу: {OLLAMA_HOST}")
    print(f"Используемая модель: {OLLAMA_MODEL}")
    print(f"Результаты будут сохранены в папку: '{OUTPUT_DIR}'")
    
    try:
        client = Client(host=OLLAMA_HOST)
        client.show(OLLAMA_MODEL) 
        print("Соединение с сервером установлено, модель найдена.")
    except Exception as e:
        print(f"\n!!! КРИТИЧЕСКАЯ ОШИБКА !!!")
        print(f"Не удалось подключиться к серверу Ollama по адресу '{OLLAMA_HOST}' или найти модель '{OLLAMA_MODEL}'.")
        print(f"Ошибка: {e}")
        print("Проверьте следующее:")
        print("1. Сервер Ollama запущен на указанной машине.")
        print("2. Адрес OLLAMA_HOST в скрипте указан верно (IP и порт).")
        print(f"3. На сервере Ollama установлена модель '{OLLAMA_MODEL}' (команда: ollama pull {OLLAMA_MODEL})")
        print("4. Если сервер на другой машине, убедитесь, что он настроен для приема внешних подключений.")
        exit()
        
    successful_generations = 0
    for i in range(NUM_INVOICES_TO_GENERATE):
        print(f"\n===== Сборка счета {i+1}/{NUM_INVOICES_TO_GENERATE} =====")
        header, table_df = generate_full_invoice(client=client, model_name=OLLAMA_MODEL, faker=faker)
        
        if header and table_df is not None and not table_df.empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"invoice_{timestamp}.xlsx"
            file_path = os.path.join(OUTPUT_DIR, filename)

            try:
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    
                    # ★★★ КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ ОШИБКИ "At least one sheet must be visible" ★★★
                    # Гарантируем, что все значения в шапке являются простыми строками.
                    # Это предотвращает сбой openpyxl, если LLM вернет сложное значение (например, список или словарь).
                    sanitized_header = {str(k): str(v) for k, v in header.items()}
                    header_df = pd.DataFrame.from_dict(sanitized_header, orient='index', columns=['Значение'])
                    header_df.index.name = "Поле"
                    header_df.to_excel(writer, sheet_name='Счет', startrow=0)

                    # Сохраняем таблицу с отступом от шапки
                    start_row_for_table = len(header) + 2 
                    table_df.to_excel(writer, sheet_name='Счет', startrow=start_row_for_table, index=False)
                
                print(f"Счет успешно собран и сохранен в файл: {file_path}")
                successful_generations += 1
            except Exception as e:
                print(f"Ошибка при сохранении Excel-файла: {e}")
            
    print(f"\n\nГенерация завершена. Успешно создано и сохранено файлов: {successful_generations}")