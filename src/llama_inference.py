import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging
logging.set_verbosity_info() 
from typing import Optional


class LlamaInference:
    def _init_(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct", 
                 device: Optional[str] = None, 
                 torch_dtype: torch.dtype = torch.float16):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        
        self.tokenizer = None
        self.model = None
        
        print(f"Initializing Llama inference with model: {model_name} on device: {self.device}")
        
        self._load_model()
    
    def _load_model(self):
        try:
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if self.device == "cuda" else {"": "cpu"},
                torch_dtype=self.torch_dtype,
                resume_download=True
            )

            if self.device == "cpu":
                self.model = self.model.to("cpu")
            else:
                self.model = self.model.to(self.device)
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def generate_text(self, 
                     input_text: str, 
                     max_length: int = 512,
                     temperature: float = 0.7,
                     top_p: float = 0.9,
                     do_sample: bool = True,
                     num_return_sequences: int = 1,
                     **kwargs) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Initialize the class first.")
        
        try:
            inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if generated_text.startswith(input_text):
                generated_text = generated_text[len(input_text):].strip()
            
            return generated_text
            
        except Exception as e:
            print(f"Error during text generation: {str(e)}")
            raise

    def generate_batch(
        self,
        input_texts: list[str],
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **kwargs
    ) -> list[str]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Initialize the class first.")
        
        try:
            inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )

            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            results = []
            for i, text in enumerate(generated_texts):
                original_prompt = input_texts[i // num_return_sequences]
                if text.startswith(original_prompt):
                    text = text[len(original_prompt):].strip()
                results.append(text)

            return results
        
        except Exception as e:
            print(f"Error during batch text generation: {str(e)}")
            raise