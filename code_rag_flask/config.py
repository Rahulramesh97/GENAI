import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_API_KEY = ""
    EMBEDDINGS_DIR = "faiss_index"

