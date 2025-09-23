import time
from typing import Optional, Dict, Any, List
import groq
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class GroqChatClient:
    def __init__(self, default_model: Optional[str] = None, default_params: Optional[Dict[str, Any]] = None):
        self.client = Groq()
        self.default_model = default_model
        self.default_params = default_params or {}

    def chat(self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None, 
        params: Optional[Dict[str, Any]] = None, 
        max_retries: int = 5) -> str:
        """
        :param prompt: Input prompt for the model.
        :param params: Optional parameters (overrides defaults).
        :param max_retries: How many times to retry on rate limits.
        :return: Model response as string.
        """
        chosen_model = model or self.default_model
        if not chosen_model:
            raise ValueError("No model provided (neither default nor per-request).")
        merged_params = {**self.default_params, **(params or {})}
        retries = 0

        while retries < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=chosen_model,
                    messages=messages,
                    **merged_params
                )
                return response.choices[0].message.content

            except groq.RateLimitError:
                wait_time = 60 + (2 ** retries) 
                print(f"Rate limit hit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                retries += 1

        raise RuntimeError("Max retries exceeded due to rate limiting.")
