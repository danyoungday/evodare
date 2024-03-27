from abc import ABC, abstractmethod, classmethod
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def the_answer_is(output: str):
    pattern = re.compile(r'The answer is: (\$?[0-9,]+)')
    match = pattern.search(output)
    if match:
        word = match.group(1)
        digits_only = re.sub(r'[^0-9]', '', word)
        return int(digits_only)
    else:
        return -1

class LanguageModel(ABC):
    def __init__(self, model_name, max_new_tokens=1024, batch_size=1):
        self.model = None
        self.model_name = model_name.replace("/", "_")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_new_tokens=max_new_tokens
        self.batch_size=batch_size

    @classmethod
    def from_pretrained(cls, model_name, device_map=None, max_new_tokens=1024, batch_size=1, quantization_config=None):
        language_model = cls(model_name, max_new_tokens=max_new_tokens, batch_size=batch_size)
        language_model.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            quantization_config=quantization_config
        )
        return language_model
    
    @classmethod
    def from_memory(cls, model_name, model, max_new_tokens=1024, batch_size=1):
        language_model = cls(model_name, max_new_tokens=max_new_tokens, batch_size=batch_size)
        language_model.model = model
        return language_model

    @abstractmethod
    def parse_prompt(self, dataset_name: str, example: dict) -> str:
        raise NotImplementedError()

    def generate(self, prompts: list[str], verbose=0) -> list[str]:
        results = []

        iterator = range(0, len(prompts), self.batch_size)
        if verbose > 0:
            iterator = tqdm(iterator, total=max(len(prompts) // self.batch_size + (len(prompts) % self.batch_size > 0), 1))

        for batch_idx in iterator:
            batch = prompts[batch_idx:min(batch_idx + self.batch_size, len(prompts))]
            tokens = self.tokenizer(batch, return_tensors="pt", padding="longest").to(self.model.device)
            with torch.no_grad():
                output = self.model.generate(**tokens,
                                            max_new_tokens=self.max_new_tokens,
                                            do_sample=False,
                                            pad_token_id=self.tokenizer.eos_token_id)
            decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            results.extend(decoded)
        assert len(results) == len(prompts)
        return results
    
    @abstractmethod
    def parse_answer(self, dataset_name: str, output: str):
        raise NotImplementedError()

class Mistral(LanguageModel):
    def __init__(self, *args, **kwargs):
        super().__init__(
            "mistralai/Mistral-7B-Instruct-v0.2",
            *args,
            **kwargs
        )
    
    def parse_prompt(self, dataset_name: str, example: dict):
        if dataset_name == "gsm8k":
            prompt = f"<s>[INST] {example['question']} [/INST]"
        elif dataset_name == "human_eval":
            prompt = f"<s>[INST] {example['prompt']} [/INST]"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return prompt
    
    def parse_answer(self, dataset_name: str, output: str):
        if dataset_name == "gsm8k":
            return the_answer_is(output)
        elif dataset_name == "human_eval":
            response = output.split("[/INST]")[1]
            return response
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

class WizardMath(LanguageModel):
    def __init__(self, *args, **kwargs):
        super().__init__(
            "WizardLM/WizardMath-7B-V1.1",
            *args,
            **kwargs
        )

    def parse_prompt(self, dataset_name: str, example: dict):
        if dataset_name == "gsm8k":
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{example['question']}\n\n### Response:"
        elif dataset_name == "human_eval":
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{example['prompt']}\n\n### Response:"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return prompt

    def parse_answer(self, dataset_name: str, output: str):
        if dataset_name == "gsm8k":
            return the_answer_is(output)
        elif dataset_name == "human_eval":
            response = output.split("Response:")[1]
            return response
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
class Hermes(LanguageModel):
    def __init__(self, *args, **kwargs):
        super().__init__(
            "NousResearch/Hermes-2-Pro-Mistral-7B",
            *args,
            **kwargs
        )

    def parse_prompt(self, dataset_name: str, example: dict):
        if dataset_name == "gsm8k":
            system = "<|im_start|>system\nYou are \"Hermes 2\", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia. Finish your response with \"The answer is:\" and then your final answer and only the final answer as a number.<|im_end|>\n"
            user = f"<|im_start|>user\n{example['question']}<|im_end|>\n"
            assistant = "<|im_start|>assistant\n"
            return system + user + assistant
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def parse_answer(self, dataset_name: str, output: str):
        if dataset_name == "gsm8k":
            return the_answer_is(output)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
class Speechless(LanguageModel):
    def __init__(self, *args, **kwargs):
        super().__init__(
            "uukuguy/speechless-code-mistral-7b-v1.0",
            *args,
            **kwargs
        )

    def parse_prompt(self, dataset_name: str, example: dict):
        if dataset_name == "gsm8k":
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. Finish your response with \"The answer is:\" and then your final answer and only the final answer as a number.\n\n### Instruction:\n{example['question']}\n\n### Response:"
            return prompt
        if dataset_name == "human_eval":
            prompt = f"You are an intelligent programming assistant.\n\n### Instruction:\n{example['prompt']}\n\n### Response:"
            return prompt
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def parse_answer(self, dataset_name: str, output: str):
        if dataset_name == "gsm8k":
            return the_answer_is(output)
        if dataset_name == "human_eval":
            response = output.split("Response:")[1]
            return response
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
