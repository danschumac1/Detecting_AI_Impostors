'''
How to run:
   python ./src/utils/prompter.py
'''
import base64
import os
import ast
import importlib
import json
from typing import List, Dict, Optional, Tuple, Type, Union
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pydantic import BaseModel
from tqdm import tqdm
import yaml
from dotenv import load_dotenv
# import torch
import openai
# from transformers import (
#     AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM)

from utils.logging_utils import MasterLogger

class QAs(BaseModel):
    question: Dict[str, str]  # Multiple inputs as a dictionary
    answer: Union[BaseModel, str]   # Allow both strings and BaseModel

class Prompter(ABC):
    """
    An abstract base class for building prompt-driven interfaces to large language models (LLMs).

    Attributes:
        llm_model (str): The model identifier (e.g., "gpt-4").
        prompt_path (Optional[str]): Path to the YAML file defining prompt structure and few-shot examples.
        prompt_headers (Dict[str, str]): Optional mapping of input keys to display labels.
        temperature (float): Temperature used for model generation.
        output_format_class (Type[BaseModel] or str): Expected output structure.
        examples (List[QAs]): Few-shot examples loaded from YAML.
        system_prompt (str): The system message for chat-based models.
        main_prompt_header (str): Optional string prepended to each prompt.
        is_structured_output (bool): Whether output should be parsed as structured JSON.
    """

    def __init__(
            
        self,
        prompt_path: Optional[str],
        prompt_headers: Dict[str, str],
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        show_prompts = False
    ):
        """
        Initializes the prompter by loading the YAML config, few-shot examples, and model schema.

        Args:
            prompt_path (Optional[str]): Path to a YAML file defining prompt structure and examples.
            prompt_headers (Dict[str, str]): Dictionary of display names for input fields.
            llm_model (str): Name or ID of the language model (default "gpt-4o-mini").
            temperature (float): Temperature for generation sampling.
        """
        self.llm_model = llm_model
        self.prompt_path = prompt_path
        self.prompt_headers = prompt_headers
        self.temperature = temperature
        self.first_print = True
        self.show_prompts = show_prompts  # Set to False to disable prompt printing
        (
            self.output_format_class,
            self.examples,
            self.system_prompt,
            self.main_prompt_header,
            self.prompt_headers,
            self.is_structured_output  # ← add this line
        ) = self._load_yaml_examples_with_model()

        self.format_examples()


    def __repr__(self) -> str:
        return f"Prompter(model={self.llm_model}, examples={len(self.examples)})"

    def _load_yaml_examples_with_model(self) -> Tuple[Type[BaseModel], List[QAs], str, str, Dict[str, str], bool]:
        """
        Loads and parses the YAML file containing system prompt, prompt headers, and examples.

        Returns:
            Tuple containing:
                - The output model class (or `str` for unstructured outputs)
                - The list of QAs examples
                - The system prompt string
                - The main prompt header
                - A dict of prompt headers
                - A bool indicating structured output mode
        """
        with open(self.prompt_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)


        meta = raw.get("__meta__", {})
        model_path = meta.get("output_model")
        if not model_path:
            raise ValueError("YAML file must contain __meta__.output_model")

        unstructured_aliases = {"simple_string", "string", "str"}
        if model_path in unstructured_aliases:
            model_class = str
            is_structured = False
        else:
            try:
                module_name, class_name = model_path.rsplit(".", 1)
                model_module = importlib.import_module(module_name)
                model_class = getattr(model_module, class_name)
                is_structured = True
            except (ValueError, ImportError, AttributeError) as e:
                raise ImportError(
                    f"Could not load output_model '{model_path}'. Either:\n"
                    f"  - Use one of: {unstructured_aliases}, or\n"
                    f"  - Provide a valid import path like 'my_module.MyModelClass'\n"
                    f"Full error: {e}"
                )

        system_prompt = raw.get("system_prompt", "You are a helpful assistant.")
        main_prompt_header = raw.get("main_prompt_header", "")
        prompt_headers = raw.get("prompt_headers", {})
        # print(raw.get("examples", []))
        examples = [
            QAs(
                question=ex["input"], 
                answer=model_class(**ex["output"]) if is_structured else ex["output"])
            for ex in raw.get("examples", [])
        ]

        return model_class, examples, system_prompt, main_prompt_header, prompt_headers, is_structured

    def format_q_as_string(self, question_dict: Dict[str, str]) -> str:
        """
        Formats the user's question fields into a single prompt string.

        Args:
            question_dict (Dict[str, str]): A mapping of input field names to values.

        Returns:
            str: A formatted string ready for prompting the LLM.
        """
        formatted_questions = "\n\n".join(
            f"{self.prompt_headers.get(key, key).upper()}: {value}" for key, value in question_dict.items()
        )

        if self.is_structured_output:
            prompt = (
                f"{formatted_questions}\n"
                f"Provide your response in JSON format using the schema below:\n"
                f"{self.output_format_class.model_json_schema()}\n"
                f"Do not include any extra text, explanations, or comments outside the JSON object."
            )
        else:
            prompt = (
                f"{formatted_questions}\n"
                f"Answer the question clearly and concisely. Do not include explanations unless asked."
            )

        return prompt

    def format_examples(self):
        """
        Applies formatting to the loaded few-shot examples using the defined prompt headers.
        """
        for qa in self.examples:
            qa.question = self.format_q_as_string(qa.question)
            if isinstance(qa.answer, BaseModel):
                qa.answer = qa.answer.model_dump_json()

    @abstractmethod
    def parse_output(self, llm_output: str):
        """
        Abstract method for parsing the raw LLM output.

        Args:
            llm_output (str): The output from the LLM.

        Returns:
            The parsed result, either a dictionary or string.
        """
        pass

    @abstractmethod
    def get_completion(self, user_inputs: Dict[str, str]) -> str:
        """
        Abstract method to submit the formatted prompt to the model and return its output.

        Args:
            user_inputs (Dict[str, str]): Inputs for the prompt.

        Returns:
            str: Raw output from the LLM.
        """
        pass

# === OpenAI Implementation ===
class OpenAIPrompter(Prompter):
    """
    Concrete implementation of the Prompter base class using OpenAI's chat completion API.
    """
    def __init__(self, llm_model="gpt-4o-mini", **kwargs):
        super().__init__(**kwargs)
        self.client = openai.Client(api_key=self._load_env())

    def _load_env(self) -> str:
        """
        Loads the OpenAI API key from a `.env` file located at `./resources/.env`.

        Returns:
            str: The API key string.

        Raises:
            ValueError: If the key is missing from the environment.
        """
        load_dotenv("./resources/.env")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(f"API Key not found. Set OPENAI_API_KEY=xxxx in ./resources/.env")
        return api_key
    
    def parse_output(self, llm_output) -> Union[str, dict]:
        """
        Parses the LLM output depending on whether structured mode is active.

        Args:
            llm_output: The response object returned from the OpenAI API.

        Returns:
            Union[str, dict]: A parsed JSON object or raw string.
        """
        content = llm_output.choices[0].message.content.strip()

        if self.is_structured_output:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                raise ValueError(f"Expected JSON but got invalid response:\n{content}")
        else:
            return content  # raw string
        
    def add_image(self, input_dict: Dict[str, str], image_path: str):
        """
        Encodes a PNG image as base64 and adds it to the input dictionary under `__image__`.

        Args:
            input_dict (Dict[str, str]): The input dictionary to augment.
            image_path (str): Path to the image file.
        """
        with open(image_path, "rb") as img:
            base64_img = base64.b64encode(img.read()).decode("utf-8")
        input_dict["__image__"] = f"data:image/png;base64,{base64_img}"


    def _build_messages(self, input_texts: Dict[str, str]):
        """
        Constructs a list of chat-style message dictionaries for OpenAI's API.

        Includes system prompt, few-shot examples, and final user input.
        Automatically handles image attachments.

        Args:
            input_texts (Dict[str, str]): The current user input.

        Returns:
            List[Dict]: The formatted message sequence.
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add few-shot examples (already pre-formatted)
        for qa in self.examples:
            messages.append(
                {"role": "user", "content": f"{self.main_prompt_header}\n{qa.question}"}
            )
            messages.append(
                {"role": "assistant", "content": qa.answer}
            )

        # Format final user input
        user_input_prompt = self.format_q_as_string(input_texts)
        if "__image__" in input_texts:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input_prompt},
                    {"type": "image_url", "image_url": {"url": input_texts["__image__"]}}
                ]
            })
        else:
            messages.append({"role": "user", "content": f"{self.main_prompt_header}\n{user_input_prompt}"})


        if self.show_prompts and self.first_print:
            self.first_print = False
            print("=" * 50)
            print("=" * 17, "EXAMPLE PROMPT", "=" * 17)
            print(json.dumps(messages, indent=4))
            print("=" * 50)

        return messages

    def get_completion(
            self, input_texts: Dict[str, str], parse=True, verbose=False) -> Union[dict, None]:
        """
        Sends a prompt to the OpenAI chat API and returns the parsed or raw response.

        Args:
            input_texts (Dict[str, str]): Dictionary of input fields for the prompt.
            parse (bool): Whether to parse the response or return raw.
            verbose (bool): Whether to print the response to console.

        Returns:
            Union[dict, None]: The parsed response, or None on failure.
        """
        input_text_str = self._build_messages(input_texts)
        completion_kwargs = {
            "model": self.llm_model,
            "messages": input_text_str,
            "temperature": self.temperature,
        }

        if self.is_structured_output:
            completion_kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**completion_kwargs)


        final_resp = self.parse_output(response) if parse else response

        if verbose:
            print("\n" + "="*60)
            print("OUTPUT FROM LLM:")
            print(json.dumps(final_resp, indent=4))
            print("="*60 + "\n")

        return [final_resp]
    
    def batch_generate(
        self,
        inputs: List[Dict[str, str]],
        max_workers: int = 10,
        verbose: bool = False,
        sleep_between: float = 0.0
    ) -> List[Union[str, dict]]:
        """
        Runs multiple prompt generations concurrently using threads.

        Args:
            inputs (List[Dict[str, str]]): List of user input dictionaries.
            max_workers (int): Max number of parallel threads.
            verbose (bool): Whether to print each output.
            sleep_between (float): Seconds to sleep between request launches.

        Returns:
            List[Union[str, dict]]: List of parsed model outputs (or error messages).
        """
        results = [None] * len(inputs)

        def task(i, input_data):
            if sleep_between > 0:
                time.sleep(sleep_between)
            try:
                result = self.get_completion(input_data, parse=True, verbose=verbose)
                return i, result
            except Exception as e:
                return i, {"error": str(e)}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(task, i, inputs[i]) for i in range(len(inputs))]
            for future in as_completed(futures):
                i, output = future.result()
                results[i] = output

        return results


# class HFPrompter(Prompter):
#     def __init__(
#         self,
#         llm_model: str,
#         max_new_tokens: int = 2000,
#         temperature: float = 0.1,
#         quantize: bool = False,
#         device_map: Union[str, int, dict] = 0,
#         torch_dtype=torch.float16,
#         **kwargs, 
#     ):
#         super().__init__(llm_model=llm_model, temperature=temperature, **kwargs)

#         # Handle device_map flexibly
#         if isinstance(device_map, int):
#             device_map = {"": device_map}
#         elif not isinstance(device_map, (str, dict)):
#             raise ValueError("device_map must be a string ('auto'), int, or dict")

#         self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.max_new_tokens = max_new_tokens
#         self.first_print = True

#         model_kwargs = {
#             "device_map": device_map,
#             "torch_dtype": torch_dtype
#         }

#         if quantize:
#             model_kwargs["quantization_config"] = BitsAndBytesConfig(
#                 load_in_8bit=True,
#                 bnb_4bit_compute_dtype=torch.float16
#             )

#         self.model = AutoModelForCausalLM.from_pretrained(llm_model, **model_kwargs)

#     def parse_output(self, generated_text: str) -> Union[str, dict]:
#         cleaned = generated_text.strip()

#         if not self.is_structured_output:
#             return cleaned

#         if cleaned.lower().startswith("object."):
#             cleaned = cleaned[len("object."):].strip()

#         try:
#             return json.loads(cleaned)
#         except json.JSONDecodeError:
#             try:
#                 return ast.literal_eval(cleaned)
#             except Exception:
#                 raise ValueError(f"Output was not valid JSON or Python literal:\n{generated_text}")

#     def _build_messages(self, input_texts: Dict[str, str]) -> List[Dict[str, str]]:
#         messages = [{"role": "system", "content": self.system_prompt}]
#         for qa in self.examples:
#             messages.append({"role": "user", "content": f"{self.main_prompt_header}\n{qa.question}"})
#             messages.append({"role": "assistant", "content": qa.answer})
#         final_prompt = self.format_q_as_string(input_texts)
#         messages.append({"role": "user", "content": f"{self.main_prompt_header}\n{final_prompt}"})

#         if self.first_print:
#             self.first_print = False
#             print("=" * 50)
#             print("=" * 17, "EXAMPLE PROMPT", "=" * 17)
#             print(json.dumps(messages, indent=4))
#             print("=" * 50)

#         return messages

#     def get_completion(self, input_texts: Union[Dict[str, str], List[Dict[str, str]]], parse=True) -> Union[dict, List[dict], str]:
#         if isinstance(input_texts, dict):
#             input_texts = [input_texts]

#         messages_list = [self._build_messages(item) for item in input_texts]

#         prompts = [
#             self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
#             for m in messages_list
#         ]

#         inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
#         input_ids = inputs["input_ids"].to(self.model.device)
#         attention_mask = inputs["attention_mask"].to(self.model.device)

#         outputs = self.model.generate(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             max_new_tokens=self.max_new_tokens,
#             return_dict_in_generate=True
#         )

#         generated_sequences = outputs.sequences
#         results = []
#         for i in range(len(generated_sequences)):
#             input_len = (input_ids[i] != self.tokenizer.pad_token_id).sum().item()
#             gen_tokens = generated_sequences[i][input_len:]
#             decoded = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
#             results.append(self.parse_output(decoded) if parse else decoded)

#         return [results[0]] if len(results) == 1 else results
