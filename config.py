# config.py
# Nơi lưu trữ tất cả các biến cấu hình quan trọng của dự án.

# Tên mô hình AI từ Hugging Face để tạo vector
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# Số chiều của vector mà mô hình trên tạo ra
VECTOR_DIMENSION = 384

# Tên của Index (cơ sở dữ liệu) trên Pinecone
PINECONE_INDEX_NAME = "so-tay-nha-in"

# Các thông số kỹ thuật cho việc tạo Index trên Pinecone
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = 'aws'
PINECONE_REGION = 'us-east-1'
