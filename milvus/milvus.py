from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer

# 1. 初始化 Embedding 模型 (会自动下载模型，约 400MB)
# 该模型支持多语言（含中文），输出维度为 384
print("正在加载 Embedding 模型...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
DIMENSION = 384 

# 2. 连接 Milvus
connections.connect("default", host="localhost", port="19530")

COLLECTION_NAME = "my_knowledge_base"

# 3. 配置集合 (Schema)
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),  # 存储原始文本
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=100) # 存储来源标签
]
schema = CollectionSchema(fields, "本地知识库演示")
collection = Collection(COLLECTION_NAME, schema)

# 4. 准备原始文本数据
raw_documents = [
    {"text": "Milvus 是一款开源的向量数据库，适用于 AI 应用。", "source": "official"},
    {"text": "向量维度代表了特征的数量，维度越高描述越细腻。", "source": "blog"},
    {"text": "HNSW 是一种基于图结构的索引算法，检索速度极快。", "source": "wiki"},
    {"text": "苹果公司发布了最新的 MacBook Pro 笔记本电脑。", "source": "news"},
    {"text": "如何使用 Python 操作向量数据库进行语义搜索？", "source": "qa"}
]

# 5. 将文本转化为向量 (Embedding)
print("正在生成向量...")
texts = [doc["text"] for doc in raw_documents]
vectors = model.encode(texts) # 关键步骤：文本转向量
sources = [doc["source"] for doc in raw_documents]

# 6. 插入数据到 Milvus
data = [vectors, texts, sources]
collection.insert(data)
collection.flush() # 确保数据持久化

# 7. 创建 HNSW 索引以加速搜索
index_params = {
    "metric_type": "L2",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 256}
}
collection.create_index(field_name="vector", index_params=index_params)
collection.load()

# 8. 执行语义检索
query_text = "我想了解向量数据库的索引算法"
print(f"\n查询文本: '{query_text}'")

# 将查询语句也转化为向量
query_vector = model.encode([query_text])

search_params = {"metric_type": "L2", "params": {"ef": 64}}
results = collection.search(
    data=query_vector, 
    anns_field="vector", 
    param=search_params, 
    limit=2, 
    output_fields=["text", "source"]
)

# 9. 展示结果
print("\n--- 检索结果 (语义相关度排序) ---")
for hits in results:
    for hit in hits:
        print(f"相似度距离: {hit.distance:.4f}")
        print(f"原始文本: {hit.entity.get('text')}")
        print(f"来源: {hit.entity.get('source')}\n")

# 释放
collection.release()