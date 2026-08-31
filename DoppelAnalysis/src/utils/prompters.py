from __future__ import annotations
import os
import json
import re
import torch
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Type, Iterable, List, Any, Union, cast
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from json_repair import repair_json
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# Specific import for Qwen 3.5 architecture compatibility
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from utils.prompt_objects import FullPrompt

# Define TypeVar bound to either a Pydantic Model or a String
T = TypeVar("T", bound=Union[BaseModel, str])

# --- BASE CLASS ---
class Prompter(ABC, Generic[T]):
    def __init__(self, model_id: str, temperature: float = 0.1):
        self.model_id = model_id
        self.temperature = temperature
        load_dotenv("./resources/.env")

    @abstractmethod
    def get_completion(
        self, 
        full_prompt: FullPrompt[T], 
        current_query: dict[str, str], 
        response_model: Type[T]
    ) -> T:
        pass

    @abstractmethod
    def get_completions_batch(
        self,
        full_prompt: FullPrompt[T],
        queries: Iterable[dict[str, str]],
        response_model: Type[T],
        batch_size: int = 4,
        **kwargs: Any
    ) -> List[T]:
        pass

    def _extract_json(self, text: str) -> Any:
        """Robustly extract and repair JSON from model output."""
        if "</think>" in text:
            text = text.split("</think>")[-1]
        
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
            
        obj = repair_json(text, return_objects=True)
        if isinstance(obj, list) and len(obj) > 0:
            obj = obj[0]
        return obj

# --- OPENAI (THREADED BATCHING) ---
class OpenAIPrompter(Prompter[T]):
    def __init__(self, model_id: str = "gpt-4o-mini", temperature: float = 0.1):
        super().__init__(model_id, temperature)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _format_input(self, data: dict[str, str]) -> str:
        return "\n".join([f"{k.upper()}: {v}" for k, v in data.items()])

    def _build_messages(self, full_prompt: FullPrompt[T], current_query: dict[str, str], response_model: Type[T]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": full_prompt.system_prompt}]
        for ex in full_prompt.examples:
            messages.append({"role": "user", "content": self._format_input(ex.user_input)})
            content = ex.assistant_output.model_dump_json() if isinstance(ex.assistant_output, BaseModel) else ex.assistant_output
            messages.append({"role": "assistant", "content": content})

        final_text = self._format_input(current_query)
        if issubclass(response_model, BaseModel):
            final_text += f"\n\nReturn JSON matching this schema:\n{json.dumps(response_model.model_json_schema())}"

        messages.append({"role": "user", "content": final_text})
        return messages

    def get_completion(self, full_prompt: FullPrompt[T], current_query: dict[str, str], response_model: Type[T]) -> T:
        messages = self._build_messages(full_prompt, current_query, response_model)
        is_pydantic = issubclass(response_model, BaseModel)
        
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages, # type: ignore
            temperature=self.temperature,
            response_format={"type": "json_object"} if is_pydantic else {"type": "text"}
        )
        
        raw_content = response.choices[0].message.content or ""
        if is_pydantic:
            return cast(T, response_model.model_validate(self._extract_json(raw_content)))
        return cast(T, raw_content)

    def get_completions_batch(self, full_prompt, queries, response_model, max_workers=8, **kwargs) -> List[T]:
        query_list = list(queries)
        results: List[T] = [cast(T, None)] * len(query_list)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.get_completion, full_prompt, q, response_model): i for i, q in enumerate(query_list)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results

# --- QWEN 3.5 ---
import os
import json
import torch
from typing import List, Type, cast
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoConfig, Qwen3_5ForCausalLM

class QwenPrompter(Prompter[T]):
    def __init__(self, model_id: str = "Qwen/Qwen3.5-27B", temperature: float = 0.7, device: str = "cuda", debug: bool = False):
        super().__init__(model_id, temperature)
        hf_token = os.getenv("HF_TOKEN")
        self.device = device
        self.debug = debug

        if self.debug:
            print(f"[DEBUG] Initializing QwenPrompter with model: {model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        full_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
        llm_config = full_config.text_config if hasattr(full_config, "text_config") else full_config

        self.model = Qwen3_5ForCausalLM.from_pretrained(
            model_id, 
            config=llm_config, 
            device_map=device, 
            torch_dtype="auto", 
            trust_remote_code=True, 
            token=hf_token
        ).eval()

    def _build_prompt(self, full_prompt: FullPrompt[T], current_query: dict[str, str], response_model: Type[T]) -> str:
        # Strict system prompt for structured output
        system_content = f"{full_prompt.system_prompt}\nRespond ONLY with a JSON object. No preamble or reasoning."
        messages = [{"role": "system", "content": system_content}]
        
        for ex in full_prompt.examples:
            messages.append({"role": "user", "content": json.dumps(ex.user_input)})
            content = ex.assistant_output.model_dump_json() if isinstance(ex.assistant_output, BaseModel) else ex.assistant_output
            messages.append({"role": "assistant", "content": content})

        user_text = json.dumps(current_query)
        if issubclass(response_model, BaseModel):
            # Use a simplified schema hint to prevent type-name confusion (e.g., writing "integer")
            props = response_model.model_json_schema().get("properties", {})
            schema_hint = {k: v.get("type") for k, v in props.items()}
            user_text += f"\n\nReturn JSON object with keys {list(schema_hint.keys())}: {json.dumps(schema_hint)}"
        
        messages.append({"role": "user", "content": user_text})
        
        # Apply the Reddit fix: enable_thinking=False stops the model from yapping in <think> tags
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False 
        )
        
        if self.debug:
            print(f"\n--- [DEBUG] PROMPT START ---\n{prompt}\n--- [DEBUG] PROMPT END ---\n")
            
        return prompt

    def get_completions_batch(self, full_prompt, queries, response_model, batch_size=4, **kwargs) -> List[T]:
        results: List[T] = []
        query_list = list(queries)
        
        for i in range(0, len(query_list), batch_size):
            chunk = query_list[i : i + batch_size]
            prompts = [self._build_prompt(full_prompt, q, response_model) for q in chunk]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            
            with torch.inference_mode():
                # Optimized parameters based on Qwen3.5 community recommendations
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=512,  # Lowered: no longer need room for long <think> blocks
                    temperature=self.temperature, 
                    top_p=0.8,           # Recommended for Qwen Instruct
                    repetition_penalty=1.05, 
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=True if self.temperature > 0 else False,
                    stop_strings=["}", "}\n", "<|im_end|>"], # Stop immediately after JSON closes
                    tokenizer=self.tokenizer
                )
            
            prompt_len = inputs.input_ids.shape[1]
            decoded = self.tokenizer.batch_decode(generated_ids[:, prompt_len:], skip_special_tokens=True)
            
            for text in decoded:
                if self.debug:
                    print(f"\n--- [DEBUG] RAW OUTPUT ---\n{text}\n--- [DEBUG] END RAW OUTPUT ---\n")

                # Secondary safety: strip thinking tags if the template flag is ignored by certain model versions
                clean_text = text.split("</think>")[-1].strip()

                if issubclass(response_model, BaseModel):
                    try:
                        # Find the LAST JSON block to avoid picking up the schema from the prompt history
                        extracted_data = self._extract_json(clean_text)
                        results.append(cast(T, response_model.model_validate(extracted_data)))
                    except Exception as e:
                        if self.debug:
                            print(f"[DEBUG] Extraction/Validation failed for: {clean_text}")
                        raise e
                else:
                    results.append(cast(T, clean_text))
                    
        return results

    def get_completion(self, full_prompt: FullPrompt[T], current_query: dict[str, str], response_model: Type[T]) -> T:
        return self.get_completions_batch(full_prompt, [current_query], response_model, batch_size=1)[0]


# --- LLAMA 3.1 ---
class LlamaPrompter(Prompter[T]):
    def __init__(self, model_id: str = "meta-llama/Llama-3.1-8B-Instruct", temperature: float = 0.1, device: str = "cuda", debug: bool = False):
        super().__init__(model_id, temperature)
        hf_token = os.getenv("HF_TOKEN")
        self.device = device
        self.debug = debug

        if self.debug:
            print(f"[DEBUG] Initializing LlamaPrompter with model: {model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map=device, 
            torch_dtype=torch.bfloat16, 
            token=hf_token
        ).eval()

    def _build_prompt(self, full_prompt: FullPrompt[T], current_query: dict[str, str], response_model: Type[T]) -> str:
        # Llama 3.1 responds well to very direct system constraints
        system_content = f"{full_prompt.system_prompt}\nRespond ONLY with a JSON object. No preamble, no explanation."
        messages = [{"role": "system", "content": system_content}]
        
        for ex in full_prompt.examples:
            messages.append({"role": "user", "content": json.dumps(ex.user_input)})
            content = ex.assistant_output.model_dump_json() if isinstance(ex.assistant_output, BaseModel) else ex.assistant_output
            messages.append({"role": "assistant", "content": content})
        
        user_text = json.dumps(current_query)
        if issubclass(response_model, BaseModel):
            # Same simplified schema trick to avoid the model writing "integer" as a string
            props = response_model.model_json_schema().get("properties", {})
            schema_hint = {k: v.get("type") for k, v in props.items()}
            user_text += f"\n\nReturn JSON: {json.dumps(schema_hint)}"
        
        messages.append({"role": "user", "content": user_text})
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        if self.debug:
            print(f"\n--- [DEBUG] LLAMA PROMPT START ---\n{prompt}\n--- [DEBUG] LLAMA PROMPT END ---\n")
            
        return prompt

    def get_completion(self, full_prompt: FullPrompt[T], current_query: dict[str, str], response_model: Type[T]) -> T:
        return self.get_completions_batch(full_prompt, [current_query], response_model, batch_size=1)[0]

    def get_completions_batch(self, full_prompt, queries, response_model, batch_size=4, **kwargs) -> List[T]:
        results: List[T] = []
        query_list = list(queries)
        
        # Llama 3.1 specific terminators
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        for i in range(0, len(query_list), batch_size):
            chunk = query_list[i : i + batch_size]
            prompts = [self._build_prompt(full_prompt, q, response_model) for q in chunk]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=512, # Llama is faster and doesn't "think", so 512 is plenty
                    temperature=self.temperature, 
                    pad_token_id=self.tokenizer.pad_token_id, 
                    eos_token_id=terminators,
                    do_sample=True if self.temperature > 0 else False,
                    # Force stop at closing brace to prevent "I hope this helps!"
                    stop_strings=["}", "}\n"], 
                    tokenizer=self.tokenizer
                )
            
            prompt_len = inputs.input_ids.shape[1]
            decoded = self.tokenizer.batch_decode(generated_ids[:, prompt_len:], skip_special_tokens=True)
            
            for text in decoded:
                if self.debug:
                    print(f"\n--- [DEBUG] LLAMA RAW OUTPUT ---\n{text}\n--- [DEBUG] END RAW OUTPUT ---\n")

                if issubclass(response_model, BaseModel):
                    try:
                        # Extract JSON (Base class handles finding { } via regex)
                        extracted_data = self._extract_json(text)
                        results.append(cast(T, response_model.model_validate(extracted_data)))
                    except Exception as e:
                        if self.debug:
                            print(f"[ERROR] Llama Validation Failed: {text}")
                        raise e
                else:
                    results.append(cast(T, text))
        return results
    