# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingCanonicalizer:
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.8):
        self.embedding_model_name = embedding_model
        self.similarity_threshold = similarity_threshold
        
        self.embedding_model = None
        
        # self._setup_embedding_model()
    
    # def _setup_embedding_model(self):
    #     try:
    #         print(f"Loading embedding model: {self.embedding_model_name}")
    #         self.embedding_model = SentenceTransformer(self.embedding_model_name)
    #         print("Embedding model loaded successfully")
    #     except Exception as e:
    #         print(f"Error loading embedding model: {e}")
    #         raise