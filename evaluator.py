import re

from datasets import load_dataset

from lm import LanguageModel

class Evaluator():

    def __init__(self):
        self.ds = load_dataset("gsm8k", "main")

    def parse_answer(self, answer):
        pattern = re.compile(r'#### (.+)$')
        matches = pattern.findall(answer)
        parsed = matches[0]
        parsed = parsed.replace(',', '')
        return int(parsed)

    def evaluate(self, lm: LanguageModel, n=-1, verbose=0):
        test_ds = self.ds["test"]
        test_ds = test_ds.map(lambda example: {"prompt": lm.parse_prompt(example["question"]), "parsed_answer": self.parse_answer(example["answer"])})
        if n != -1:
            test_ds = test_ds[:n]
        prompts = test_ds["prompt"]

        results = lm.generate(prompts, verbose=verbose)
        parsed_results = [lm.parse_answer(r) for r in results]

        if verbose > 1:
            print(results)
            print(test_ds["answer"])
        if verbose > 0:
            print(list(test_ds["parsed_answer"]), parsed_results)

        correct = 0
        for answer, result in zip(test_ds["parsed_answer"], parsed_results):
            if answer == result:
                correct += 1
        return correct / n