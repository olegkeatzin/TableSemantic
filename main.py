from models import OllamaLLM , OllamaEmbedding

llm = OllamaLLM(host = 'http://100.74.62.22:11434', model="gemma3:1b")
print(llm.generate("Сколько будет 2*2?"))

embedder = OllamaEmbedding(host = 'http://100.74.62.22:11434', model="embeddinggemma")
print(embedder.embed("Физический смысл интеграла по времени от функции скорости равенствуется перемещению тела."))