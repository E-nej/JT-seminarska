from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import time
import torch
import openai
from google import genai
#import ollama
from google.genai import types
from .consts import OPENAI_API_KEY, GEMINI_API_KEY, arapaho_morphology

valid_models = [
    "local",
    "gpt-3.5-turbo-1106",
    "gpt-4o-2024-08-06",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "gpt-4o-mini-2024-07-18",
    "gemini-3.1-flash-lite-preview",
    "qwen2.5:7b", 
]

class LLMWrapper:
    def __call__(self, messages):
        raise NotImplementedError
    

class LlamaCppWrapper(LLMWrapper):
    """Local llama.cpp server via OpenAI-compatible API."""
    def __init__(self, host="127.0.0.1", port=8080):
        self.client = openai.OpenAI(
            api_key="dummy",
            base_url=f"http://{host}:{port}/v1"
        )

    def __call__(self, messages) -> str:
        response = self.client.chat.completions.create(
            model="local",
            messages=messages,
            stream=False,
            temperature=0.0,
            top_p=1.0,
        )
        return response.choices[0].message.content


class ChatGPTWrapper(LLMWrapper):
    def __init__(self, model_id):
        if not OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Export it before running GPT-based pipelines."
            )
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model_id = model_id

    def __call__(self, messages) -> str:
        max_attempts = 3
        backoff_seconds = 2

        for attempt in range(1, max_attempts + 1):
            try:
                stream = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    stream=True,
                    temperature=0.0,
                    top_p=1.0,
                )
                content = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        content += chunk.choices[0].delta.content
                return content

            except openai.RateLimitError as exc:
                error_body = getattr(exc, "body", {}) or {}
                error_obj = error_body.get("error", {}) if isinstance(error_body, dict) else {}
                error_code = error_obj.get("code") or error_obj.get("type", "")
                if error_code == "insufficient_quota":
                    raise RuntimeError(
                        "OpenAI API request failed: insufficient quota for this API key. "
                        "Add billing/credits or switch to a non-OpenAI model id in --llm."
                    ) from exc
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"OpenAI API request failed after {max_attempts} retries due to rate limiting "
                        f"(error body: {error_body}). "
                        "Wait a moment and try again, or use a different --llm."
                    ) from exc
                print(f"Rate limited (attempt {attempt}/{max_attempts}), retrying in {backoff_seconds * attempt}s...")
                time.sleep(backoff_seconds * attempt)
            except Exception as exc:
                raise RuntimeError(f"OpenAI API request failed: {exc}") from exc

        raise RuntimeError("OpenAI API request failed after retries.")
    
    
class HFWrapper(LLMWrapper):
    def __init__(self, model_id):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.model_id == "mistralai/Mixtral-8x7B-Instruct-v0.1":
            self.quantization_config = quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    
    def __call__(self, messages):
        if self.model_id == "mistralai/Mixtral-8x7B-Instruct-v0.1" or self.model_id == "mistralai/Mistral-7B-Instruct-v0.2":
            if len(messages[1]["content"]) > 300000:
                pos = messages[1]["content"].find("Please help me translate the following sentence from ")
                messages[1]["content"] = f"""\
Here is a grammar book of Arapaho:

{arapaho_morphology}

""" + messages[1]["content"][pos:]
                
            messages = [
                {"role": "user", "content": messages[0]["content"] + messages[1]["content"]},
            ] + messages[2:]
        tokenized_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        # tokenized_chat = tokenized_chat 
        outputs = self.model.generate(
            tokenized_chat.to(self.model.device),
            max_new_tokens=32000,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0][len(tokenized_chat[0]):], skip_special_tokens=True)


class GeminiWrapper(LLMWrapper):
    def __init__(self, model_id):
        if not GEMINI_API_KEY:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Export it before running Gemini-based pipelines."
            )
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_id = model_id
    
    def __call__(self, messages) -> str:
        system_instruction = None
        conversation = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                role = "user" if msg["role"] == "user" else "model"
                conversation.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
            top_p=1.0,
        )

        max_attempts = 3
        backoff_seconds = 2

        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=conversation,
                    config=config,
                )
                return response.text
            except Exception as exc:
                if attempt == max_attempts:
                    raise RuntimeError(f"Gemini API request failed after {max_attempts} retries: {exc}") from exc
                print(f"Gemini API error (attempt {attempt}/{max_attempts}), retrying in {backoff_seconds * attempt}s...")
                time.sleep(backoff_seconds * attempt)

        raise RuntimeError("Gemini API request failed after retries.")

class OllamaWrapper(LLMWrapper):
    def __init__(self, model_id):
        self.model_id = model_id
    
    def __call__(self, messages) -> str:
        max_attempts = 3
        backoff_seconds = 2

        for attempt in range(1, max_attempts + 1):
            try:
                response = ollama.chat(
                    model=self.model_id,
                    messages=messages,
                    options={
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "num_ctx": 8192,  # 8k fits comfortably on your 32GB RAM
                    }
                )
                return response['message']['content']
            except Exception as exc:
                if attempt == max_attempts:
                    raise RuntimeError(f"Ollama request failed after {max_attempts} retries: {exc}") from exc
                print(f"Ollama error (attempt {attempt}/{max_attempts}), retrying in {backoff_seconds * attempt}s...")
                time.sleep(backoff_seconds * attempt)

        raise RuntimeError("Ollama request failed after retries.")

def get_llm_wrapper(model_id) -> LLMWrapper:
    if model_id == "local":
        return LlamaCppWrapper()
    elif "gpt" in model_id:
        return ChatGPTWrapper(model_id)
    elif "gemini" in model_id:
        return GeminiWrapper(model_id)
    elif ":" in model_id and "/" not in model_id:
        # Ollama model format is "name:tag" (e.g. qwen2.5:7b, llama3.1:8b)
        # HuggingFace uses "org/model" — no colon without slash
        return OllamaWrapper(model_id)
    else:
        return HFWrapper(model_id)