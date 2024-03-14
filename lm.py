from abc import ABC, abstractmethod
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

class LanguageModel(ABC):
    def __init__(self, model_name, device_map=None, max_new_tokens=1024, batch_size=1, quantization_config=None):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            quantization_config=quantization_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_new_tokens = max_new_tokens

        self.batch_size=batch_size

    @abstractmethod
    def parse_prompt(self, question: str) -> str:
        pass

    def generate(self, prompts: list[str], verbose=0) -> list[str]:
        results = []

        iterator = range(0, len(prompts), self.batch_size)
        if verbose > 0:
            iterator = tqdm(iterator, total=max(len(prompts) // self.batch_size, 1))

        for batch_idx in iterator:
            batch = prompts[batch_idx:min(batch_idx + self.batch_size, len(prompts))]
            tokens = self.tokenizer(batch, return_tensors="pt", padding="max_length").to(self.model.device)
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
    def parse_answer(self, answer: str):
        pass

class MetaMathLM(LanguageModel):
    def __init__(self, device_map=None, max_new_tokens=1024, batch_size=1, quantization_config=None):
        super().__init__(
            "meta-math/MetaMath-Mistral-7B",
            device_map=device_map,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            quantization_config=quantization_config
        )
    
    def parse_prompt(self, question: str):
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response: Let’s think step by step."
        return prompt

    def parse_answer(self, answer: str):
        pattern = re.compile(r'The answer is: (.+)$')
        match = pattern.search(answer)
        if not match:
            raise ValueError("No answer found")
        parsed = match.group(1)
        parsed = parsed.replace(',', '')
        return int(parsed)
    
class WizardMathLM(LanguageModel):
    def __init__(self, device_map=None, max_new_tokens=1024, batch_size=1, quantization_config=None):
        super().__init__(
            "WizardLM/WizardMath-7B-V1.0",
            device_map=device_map,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            quantization_config=quantization_config
        )

    def parse_prompt(self, model_name: str, example: dict):
        if model_name == "gsm8k":
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{example['question']}\n\n### Response: Let’s think step by step."
        elif model_name == "allenai/ai2_arc":
            instruction = example["question"]
            choices = example["choices"]
            for label, text in zip(choices["label"], choices["text"]):
                instruction += "\n{" + label + ": " + text + "}"
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
            print(repr(prompt))
            assert False
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return prompt
    
    def parse_answer(self, answer: str):
        pattern = re.compile(r'The answer is: (.+)')
        match = pattern.search(answer)
        if not match:
            raise ValueError("No answer found")
        parsed = match.group(1)
        parsed = parsed.replace(',', '')
        parsed = parsed.replace('.', '')
        return int(parsed)