from typing import Dict, Any, Optional
import pandas as pd

from .triplet_extraction import LLamaTripletExtractor
from .schema_deifnition import SchemaDefinition
from .embedding_canonicalization import EmbeddingCanonicalizer


class KnowledgeGraphPipeline:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
        self.triplet_extractor = LLamaTripletExtractor()
        
        # self.schema_definer = SchemaDefinition()
        
        # self.embedding_canonicalizer = EmbeddingCanonicalizer()
        
        print("Knowledge Graph Pipeline initialized")
    
    def run_pipeline(self, 
                    input_path: str, 
                    output_dir: Optional[str] = "results.json") -> Dict[str, Any]:
        try:
            print("Step 1: Loading data")
            data = pd.read_csv(input_path, header=None, names=["id", "sentence"])

            print("Step 2: Extracting triplets")
            triplets = self.triplet_extractor.extract_batch(
                data, 
                batch_size=self.config.get("batch_size", 4)
            )
            
            triplets.to_excel("results.xlsx", index=False)
            
            # Step 3: Clean and normalize triplets
            print("Step 3: Relationships descriptions from triplets")
            
            # Step 4: Embed and canonicalize triplets
            print("Step 4: Embedding and canonicalizing triplets")
            
            print("Pipeline completed successfully")
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "output_dir": output_dir
            }