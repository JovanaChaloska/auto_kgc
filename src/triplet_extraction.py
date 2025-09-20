from typing import List, Dict, Any, Optional, Tuple
import re
import pandas as pd
import json

from .llama_inference import LlamaInference
from .prompts import triplets_extraction_prompt


class LLamaTripletExtractor:
    def __init__(self):
        self.llama_extractor = LlamaInference()
        self.prompt = triplets_extraction_prompt
    
    # def _parse_triplets(self, generated_text: str) -> List[Tuple[str, str, str]]:
    #     """Llama returns the entire system prompt that also contains list of triplets in the examples plus the resul.
    #         can't split the string on 'assistant' cause it might be included in the result entities
    #         so it returns the last occurence of list of triplet tuples in the string"""
    #     clean_text = generated_text.replace("\n", "")

    #     pattern = r"\[\s*(\(\s*[^();]+\s*(?:;\s*[^();]+\s*)+\))(?:\s*,\s*\(\s*[^();]+\s*(?:;\s*[^();]+\s*)+\))*\s*\]"

    #     matches = re.findall(pattern, clean_text)
    #     matches = [m.group(0) for m in re.finditer(pattern, clean_text)]
        
    #     if matches:
    #         return matches[-1]
    #     else:
    #         return "[]"
        
    def _extract_structures(self, text: str):
        """Extracts all top-level [ ... ] blocks from text."""
        results = []
        i = 0
        while i < len(text):
            if text[i] == "[":
                depth = 1
                j = i + 1
                while j < len(text) and depth > 0:
                    if text[j] == "[":
                        depth += 1
                    elif text[j] == "]":
                        depth -= 1
                    j += 1
                if depth == 0:  # found matching ]
                    results.append(text[i:j])
                    i = j
                    continue
            i += 1
        return results


    def _parse_block(self, block: str):
        """Extracts all top-level ( ... ) tuples inside a [ ... ] block."""
        assert block[0] == "[" and block[-1] == "]"
        inner = block[1:-1].strip()
        tuples = []
        i = 0
        while i < len(inner):
            if inner[i] == "(":
                depth = 1
                j = i + 1
                while j < len(inner) and depth > 0:
                    if inner[j] == "(":
                        depth += 1
                    elif inner[j] == ")":
                        depth -= 1
                    j += 1
                tuples.append(inner[i:j])
                i = j
                continue
            i += 1
        return tuples


    def _parse_triplets(self, generated_text: str):
        """Replacement for the regex-based function: finds all [ ... ] blocks and extracts full matches."""
        clean_text = generated_text.replace("\n", "")
        matches = []
        for block in self._extract_structures(clean_text):
            tuples = self._parse_block(block)
            if tuples:  # only keep if it matches the expected ( ... ) pattern
                matches.append(block)  # mimic regex: return full [ ... ] match
        if matches:
            return matches[-1]
        else:
            return "[]"
    
    def extract_triplets(self, 
                        sentence: str, 
                        context: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            prompt = self.prompt.format(sentence=sentence)
            result = self.llama_extractor.generate_text(prompt)
            raw_triplets = self._parse_triplets(result)
            
            print(f"Extracted {len(raw_triplets)} triplets from sentence")
            return raw_triplets
            
        except Exception as e:
            print(f"Error extracting triplets: {e}")
            return []
    
    def extract_batch(self, 
                     df: pd.DataFrame, 
                     batch_size: int = 4) -> pd.DataFrame:
        df_local = df.copy()
        df_local["sentence_length"] = df_local["sentence"].str.len()
        df_local = df_local.sort_values("sentence_length").reset_index(drop=True)

        print(df_local.head())

        all_triplets = []
        i=0
        # for i in range(0, len(df), batch_size):
        #     batch = df.iloc[i:i + batch_size]
        #     sentences = batch["sentence"].tolist()

        #     triplets = self.llama_extractor.generate_batch(sentences)
        #     triplets=[self._parse_triplets(item) for item in triplets]
            
        #     all_triplets.append(triplets)
        #     print(f"Processed batch {i//batch_size + 1}/{(len(sentences) + batch_size - 1)//batch_size}")

        #dynamic batches per sentence length
        try:
            while i < len(df_local):
                batch = [df_local.iloc[i]]
                base_len = df_local.iloc[i]["sentence_length"]
                j = i + 1

                while j < len(df_local) and len(batch) < batch_size:
                    next_len = df_local.iloc[j]["sentence_length"]
                    if next_len - base_len > 15:   # stop condition
                        break
                    batch.append(df_local.iloc[j])
                    j += 1

                batch_df = pd.DataFrame(batch)
                sentences = batch_df["sentence"].tolist()
                sentences = [self.prompt.format(sentence=item) for item in sentences]
                triplets = self.llama_extractor.generate_batch(sentences)
                triplets = [self._parse_triplets(item) for item in triplets]
                all_triplets.extend(triplets)
                tmp_dict = {'triplets': all_triplets}
                with open('triplets.json', "w") as f:
                    json.dump(tmp_dict, f, indent=2)

                print(f"Processed batch {len(all_triplets)}/{len(df_local)} sentences")
                i = j if j > i else i + 1

            df_local["triplets"] = all_triplets
        
            return df_local
        except Exception as e:
            print(f'failed with {e}')
            progress_df = df_local.iloc[:len(all_triplets)].copy()
            progress_df.drop(columns="sentence_length", axis=1, inplace=True)
            progress_df["triplets"] = all_triplets

            progress_df.to_excel("partial_results.xlsx", index=False)

            print("Batch processing failed early")


