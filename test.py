import psutil
import copy
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from candidate import Candidate

if __name__ == "__main__":

    math_model = AutoModelForCausalLM.from_pretrained("WizardLM/WizardMath-7B-V1.1")
    math_tokenizer = AutoTokenizer.from_pretrained("WizardLM/WizardMath-7B-V1.1")

    code_model = AutoModelForCausalLM.from_pretrained("uukuguy/speechless-code-mistral-7b-v1.0")

    tokens = math_tokenizer("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is 2+2?\n\n### Response:", return_tensors="pt")

    output = math_model.generate(**tokens, max_new_tokens=1024)
    print("Original Math Output" + "".join(["-" for _ in range(80)]))
    print(math_tokenizer.decode(output[0], skip_special_tokens=True))

    print('Starting RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    for layer in tqdm(range(32)):
        base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        n_params = sum(p.numel() for p in base_model.model.layers[layer].parameters())

        base_layer = copy.deepcopy(base_model.model.layers[layer])
        del base_model
        
        p = 0.9
        candidate = Candidate(layer, n_params, p)
        candidate.random_init()

        candidate.merge_model(base_layer, math_model, code_model)

        del base_layer

        output = math_model.generate(**tokens, max_new_tokens=1024)
        print(f"Merged Layer {layer} Output" + "".join(["-" for _ in range(80)]))
        print(math_tokenizer.decode(output[0], skip_special_tokens=True))
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
