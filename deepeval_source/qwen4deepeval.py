import os
import json
from typing import Optional, List, Dict, Any
from openai import OpenAI

from deepeval.models import DeepEvalBaseLLM


class CustomOpenAIQwen(DeepEvalBaseLLM):
    def __init__(
        self,
        model_name: str = "deepseek-r1",
        api_key: str = "sk-bc733bafbe7242e8bf13a057ac05adb0",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        enable_thinking: bool = False,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        
        # Use provided API key or fall back to environment variable
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "API key is required. Set DASHSCOPE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def load_model(self):
        # For OpenAI-compatible APIs, no model loading is needed
        return self.client

    def generate(self, prompt: str) -> str:
        try:
            messages = [{"role": "user", "content": prompt},{"role": "system", "content": "Please return a JSON object"}]
            
            extra_body = {}
            if not self.enable_thinking:
                extra_body["enable_thinking"] = False
            
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                extra_body=extra_body if extra_body else None,
                response_format={"type": "json_object"}
            )
            
            response = completion.choices[0].message.content
            print(f"Response: {response}")
            return response
            # if "JSON" in prompt.upper() or "SCHEMA" in prompt.upper() or "EVALUATE" in prompt.upper():
            #     # Try to extract JSON from the response
            #     try:
            #         # Look for JSON in the response
            #         import re
            #         json_match = re.search(r'\{.*\}', response, re.DOTALL)
            #         if json_match:
            #             json_str = json_match.group(0)
            #             # Validate JSON
            #             json.loads(json_str)
            #             return json_str
            #         else:
            #             # If no JSON found, return a default valid JSON
            #             return '{"score": 0.5, "reason": "Default evaluation"}'
            #     except:
            #         # If JSON extraction fails, return default
            #         return '{"score": 0.5, "reason": "Default evaluation"}'
            
            # return response
            
        except Exception as e:
            error_msg = str(e)
            if "model_not_found" in error_msg or "does not exist" in error_msg:
                print(f"Model '{self.model_name}' not found. Available models on DashScope:")
                print("- qwen-plus")
                print("- qwen-turbo") 
                print("- qwen-max")
                print("- qwen-max-longcontext")
                print("- qwen2.5-72b-instruct")
                print("- qwen2.5-32b-instruct")
                print("- qwen2.5-14b-instruct")
                print("- qwen2.5-7b-instruct")
                return f"Error: Model '{self.model_name}' not available. Please use one of the listed models."
            else:
                print(f"Error generating response: {e}")
                return f"Error: {str(e)}"

    async def a_generate(self, prompt: str) -> str:
        # For simplicity, using synchronous version
        # In production, you might want to use async OpenAI client
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"{self.model_name}"


if __name__ == "__main__":
    # Example usage with OpenAI-compatible API
    try:
        # Option 1: Using OpenAI-compatible API (recommended for most use cases)
        # Use one of the available models on DashScope
        custom_llm = CustomOpenAIQwen(
            model_name="qwen-plus",  # or "qwen-turbo", "qwen-max", etc.
            max_tokens=1000,
            temperature=0.7,
            enable_thinking=False
        )
        print("Using OpenAI-compatible API...")
        
    except Exception as e:
        print(f"OpenAI API setup failed: {e}")
    
    # Test the model
    try:
        response = custom_llm.generate("Hello! Please introduce yourself briefly.")
        print(f"Model: {custom_llm.get_model_name()}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Generation failed: {e}")