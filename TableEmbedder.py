from models import OllamaLLM, OllamaEmbedding
import numpy as np
from numpy.linalg import norm

class EmbSerial:
    def __init__(self, embedder: OllamaEmbedding, llm: OllamaLLM, name: str, content: list):
        self.name = name
        self.content = content
        self.embedder = embedder
        self.llm = llm

        #Обработка столбца
        print(f'Обработка столбца {self.name}...')
        self.description = self.get_description(self.name, self.content)
        self.embedding = self.get_embedding(self.description)
        print('Столбец обработан!')
    
    def get_description(self, name: str, content: list):
        # prompt = f"""
        # Дай краткое описание для столбца '{name}', в котором содержатся следующие данные: {content[:5]}. 
        # Описание должно описывать возможное cодержание СТОЛБЦА и расшифровку его названия, для того
        # чтобы по этому описанию можно было различать разные столбцы таблицы.
        # Если столбце описывает что-то конкретное, попытайся думать шире, ведь в этом столбце могут быть более разнообразные данные.
        # Выведи описание и ничего больше
        # """

        prompt = f'''
        Дай краткое описания для столбца таблицы с названием '{name}'. Если по названию не понятно, что это за столбец,
        то попробуй угадать, что это может быть за столбец, основываясь на примере его содержимого: {content[:5]}.
        Описание должно быть универсальным, чтобы подходить для любых значений в этом столбце.
        Если столбец описывает что-то конкретное, попытайся думать шире, ведь в этом столбце могут быть более разнообразные данные.
        Выведи только описание и ничего больше.
        '''
        # description = self.llm.generate(prompt)
        description = f'{self.name}: {self.llm.generate(prompt)}'
        # description = self.name
        return description
    
    def get_embedding(self, description: str):
        embedding = self.embedder.embed(description)
        return embedding[0]
    
    def __sub__(self, other):
        """
        Переопределяет оператор вычитания (-) для вычисления косинусного расстояния
        между эмбеддингами двух экземпляров EmbSerial.
        """
        # Проверяем, что второй операнд тоже является экземпляром нашего класса
        if not isinstance(other, EmbSerial):
            return NotImplemented

        # Векторы эмбеддингов
        vec1 = self.embedding
        vec2 = other.embedding

        # Вычисляем косинусное сходство
        # Формула: (A • B) / (||A|| * ||B||)
        cosine_similarity = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
        
        # Возвращаем косинусное расстояние
        cosine_distance = 1 - cosine_similarity
        
        return cosine_distance