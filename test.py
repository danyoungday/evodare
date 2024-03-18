import torch
from transformers import BitsAndBytesConfig

from lm import WizardLM
from evaluator import ARCEvaluator
import numpy as np
from bitarray import bitarray

if __name__ == "__main__":

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    # lm = WizardLM(
    #     "WizardLM/WizardLM-7B-V1.0",
    #     device_map="auto",
    #     batch_size=2,
    #     quantization_config=nf4_config
    # )
    # evaluator = ARCEvaluator()
    # print(evaluator.evaluate(lm, n=2, verbose=2))

    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # model = AutoModelForCausalLM.from_pretrained("WizardLM/WizardLM-7B-V1.0", quantization_config=nf4_config, device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained("WizardLM/WizardLM-7B-V1.0")
    # instruction = "What is the capital of France?"
    # text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
    # tokens = tokenizer(text, return_tensors="pt").to(model.device)
    # with torch.no_grad():
    #     output = model.generate(**tokens, max_new_tokens=1024, do_sample=False)
    # decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
    # print(decoded)