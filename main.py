from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams

# Qdrantクライアントの初期化
client = QdrantClient(host='localhost', port=6333)

# コレクションの作成
client.recreate_collection(
    collection_name='my_collection',
    vectors_config=VectorParams(
        size=128,  # ベクトルの次元数
        distance='Cosine'  # 距離計算のメトリック
    )
)

# データの追加
points = [
    PointStruct(id=1, vector=[0.1] * 128, payload={"key": "value1"}),
    PointStruct(id=2, vector=[0.2] * 128, payload={"key": "value2"}),
    # 他のポイントも追加可能
]

client.upsert(
    collection_name='my_collection',
    points=points
)

# データの検索
query_vector = [0.1] * 128
search_result = client.search(
    collection_name='my_collection',
    query_vector=query_vector,
    limit=5  # 上位5件を取得
)

print(search_result)