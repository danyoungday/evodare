import torch
from transformers import BitsAndBytesConfig

from lm import WizardMath, Hermes, Speechless
from evaluator import GSMEvaluator

if __name__ == "__main__":

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    evaluator = GSMEvaluator()
    # math_lm = WizardMath(
    #     device_map="auto",
    #     batch_size=4,
    #     quantization_config=nf4_config
    # )
    # print(evaluator.evaluate(math_lm, n=30, verbose=1))
    # del math_lm
    # torch.cuda.empty_cache()

    code_lm = Speechless(
        device_map="auto",
        batch_size=4,
        quantization_config=nf4_config
    )
    print(evaluator.evaluate(code_lm, n=10, verbose=1))
