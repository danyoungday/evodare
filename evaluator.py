import copy
import re
from abc import ABC, abstractmethod
import json

from tqdm import tqdm

from datasets import load_dataset
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness
from evalplus.sanitize import sanitize

from lm import LanguageModel

class Evaluator(ABC):

    def __init__(self, dataset_name, dataset):
        self.dataset_name = dataset_name
        self.ds = dataset

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
    
class HumanEvalEvaluator(Evaluator):
    def __init__(self, num_samples_per_task=200):
        super().__init__("human_eval", None)
        self.problems = read_problems()
        self.num_samples_per_task=num_samples_per_task

    def generate_one_completion(self, lm: LanguageModel, problem: dict):
        processed_prompt = lm.parse_prompt(self.dataset_name, problem)
        result = lm.generate([processed_prompt])[0]
        parsed_answer = lm.parse_answer(self.dataset_name, result)
        sanitized_answer = sanitize(parsed_answer, entry_point=problem["entry_point"]).strip()
        return sanitized_answer

    def evaluate(self, lm: LanguageModel, n=-1, verbose=0):
        # Subset problems if n > 0
        if n > 0:
            keys = [f"HumanEval/{i}" for i in range(n)]
            problem_set = {k: self.problems[k] for k in keys}
        else:
            problem_set = copy.deepcopy(self.problems)

        # Generate responses
        samples = [
            dict(task_id=task_id, completion=self.generate_one_completion(lm, problem_set[task_id]))
            for task_id in tqdm(problem_set)
            for _ in range(self.num_samples_per_task)
        ]
        write_jsonl(f"samples_{lm.model_name}.jsonl", samples)

        # Take everything in problem_set and move them to a list where the key is "task_id"
        for key in keys:
            problem_set[key]["task_id"] = key
        problem_set = list(problem_set.values())
        with open(f"problems_{lm.model_name}.jsonl", "w") as f:
            for problem in problem_set:
                f.write(json.dumps(problem) + "\n")

        # Run HumanEval
        results = evaluate_functional_correctness(f"samples_{lm.model_name}.jsonl", problem_file=f"problems_{lm.model_name}.jsonl")
        return results["pass@1"]

    def parse_answer(self, example):
        pass
