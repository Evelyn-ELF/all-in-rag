from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 配置与存储时相同的嵌入模型（检索时需对 query 重新向量化）
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 2. 从本地加载已持久化的索引
persist_path = "./llamaindex_index_store"
storage_context = StorageContext.from_defaults(persist_dir=persist_path)
index = load_index_from_storage(storage_context)
print(f"LlamaIndex 索引已从 {persist_path} 加载")

# 3. 构建检索器并执行相似性搜索
retriever = index.as_retriever(similarity_top_k=1)
query = "LlamaIndex是做什么的？"
results = retriever.retrieve(query)

print(f"\n查询: '{query}'")
print("相似度最高的文档:")
for node in results:
    print(f"- {node.text}")
