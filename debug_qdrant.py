from qdrant_client import QdrantClient
print("Has search?", hasattr(QdrantClient, 'search'))
client = QdrantClient(host="localhost", port=6333)
print("Instance has search?", hasattr(client, 'search'))
print(dir(client))
