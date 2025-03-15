from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Filter, FieldCondition, MatchValue, SearchRequest
from transformers import BertModel, BertTokenizer
import torch

# BERTモデルとトークナイザーの初期化
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def text_to_vector(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

# Qdrantクライアントの初期化
client = QdrantClient(host='localhost', port=6333)

# コレクションの作成
client.recreate_collection(
    collection_name='my_collection',
    vectors_config=VectorParams(
        size=768,  # BERTの出力ベクトルの次元数
        distance='Cosine'  # 距離計算のメトリック
    )
)

# データの追加
texts = [
    "This is a long text value for testing purposes.",
    "Another long text value for testing."
]
points = []
for i, text in enumerate(texts):
    sentences = text.split('. ')  # 文節ごとに区切る
    for j, sentence in enumerate(sentences):
        points.append(PointStruct(id=i*100+j, vector=text_to_vector(sentence), payload={"key": f"file{i+1}.txt", "sentence": sentence}))

client.upsert(
    collection_name='my_collection',
    points=points
)

# データの検索
query_text = "This is a long text value for testing purposes."
query_vector = text_to_vector(query_text)
search_request = SearchRequest(
    vector=query_vector,
    limit=5,  # 上位5件を取得
    filter=Filter(
        must=[
            FieldCondition(
                key="key",
                match=MatchValue(value="file1.txt")
            )
        ]
    )
)
search_result = client.search(
    collection_name='my_collection',
    #request=search_request,
    query_vector=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="key",
                match=MatchValue(value="file1.txt")
            )
        ]
    )
)

print(search_result)