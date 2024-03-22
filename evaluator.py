import re
from abc import ABC, abstractmethod

from datasets import load_dataset

from lm import LanguageModel

class Evaluator(ABC):

    def __init__(self, dataset_name, dataset):
        self.dataset_name = dataset_name
        self.ds = dataset

    @abstractmethod
    def parse_answer(self, answer: str):
        pass

    def evaluate(self, lm: LanguageModel, n=-1, verbose=0):
        test_ds = self.ds["test"]
        test_ds = test_ds.map(lambda example: {"prompt": lm.parse_prompt(self.dataset_name, example), "parsed_answer": self.parse_answer(example)})
        if n != -1:
            test_ds = test_ds[:n]
        prompts = test_ds["prompt"]
        results = lm.generate(prompts, verbose=verbose)
        parsed_results = [lm.parse_answer(self.dataset_name, r) for r in results]

        if verbose > 1:
            print(results)
        if verbose > 0:
            print("Real answers:", test_ds["parsed_answer"])
            print("Predicted answers:", parsed_results)

        correct = 0
        for i, zipped in enumerate(zip(test_ds["parsed_answer"], parsed_results)):
            answer, result = zipped
            if answer == result:
                correct += 1
            elif verbose > 0:
                print(f"Wrong answer {result} (should be {answer})")
                print(repr(results[i]))
        return correct / n
    
class GSMEvaluator(Evaluator):
    def __init__(self):
        super().__init__("gsm8k", load_dataset("gsm8k", "main"))

    def parse_answer(self, example):
        pattern = re.compile(r'#### (.+)$')
        matches = pattern.findall(example["answer"])
        parsed = matches[0]
        parsed = parsed.replace(',', '')
        return int(parsed)
    
class ARCEvaluator(Evaluator):
    def __init__(self):
        super().__init__("allenai/ai2_arc", load_dataset("allenai/ai2_arc", "ARC-Easy"))

    def parse_answer(self, example):
        return example["answerKey"][0]    
