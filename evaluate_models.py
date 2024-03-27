import torch
import pandas as pd
from transformers import BitsAndBytesConfig

from lm import LanguageModel, WizardMath, Hermes, Speechless, Mistral
from evaluator import Evaluator, GSMEvaluator, HumanEvalEvaluator

def evaluate_model(lm: LanguageModel, evaluators: list[Evaluator], n=100, verbose=0):
    results = {}
    for evaluator in evaluators:
        results[evaluator.dataset_name] = evaluator.evaluate(lm, n=n, verbose=verbose)
    return results

if __name__ == "__main__":

    n = 10
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    gsm_evaluator = GSMEvaluator()
    human_eval_evaluator = HumanEvalEvaluator(num_samples_per_task=1)
    evaluators = [gsm_evaluator, human_eval_evaluator]
    results = []

    # lm = Mistral(
    #     device_map="auto",
    #     batch_size=4,
    #     quantization_config=nf4_config
    # )
    # results.append(evaluate_model(lm, evaluators, n, 1))
    # del lm
    # torch.cuda.empty_cache()

    math_lm = WizardMath(
        device_map="auto",
        batch_size=4,
        quantization_config=nf4_config
    )
    results.append(evaluate_model(math_lm, evaluators, n, 1))
    del math_lm
    torch.cuda.empty_cache()

    code_lm = Speechless(
        device_map="auto",
        batch_size=4,
        quantization_config=nf4_config
    )
    results.append(evaluate_model(code_lm, evaluators, n, 1))

    df = pd.DataFrame(results)
    df.to_csv("results.csv")