import torch
from transformers import BitsAndBytesConfig

from lm import MetaMathLM, WizardMathLM
from evaluator import ARCEvaluator

if __name__ == "__main__":

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    lm = WizardMathLM(
        device_map="auto",
        batch_size=2,
        quantization_config=nf4_config
    )
    evaluator = ARCEvaluator()
    print(evaluator.evaluate(lm, n=4, verbose=2))