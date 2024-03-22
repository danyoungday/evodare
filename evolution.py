"""
PyTorch implementation of NSGA-II.
"""

import random
import shutil
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

import nsga2_utils

class TorchPrescriptor():
    """
    Handles prescriptor candidate evolution
    """
    def __init__(self,
                 pop_size: int,
                 n_generations: int,
                 p_mutation: float):
        
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.p_mutation = p_mutation

    def _evaluate_candidates(self, candidates: list[Candidate]):
        """
        Calls prescribe and predict on candidates and assigns their metrics to the results.
        """
        for candidate in candidates:
            context_actions_df = self._prescribe(candidate)
            eluc_df, change_df = self.predict_metrics(context_actions_df)
            candidate.metrics = (eluc_df["ELUC"].mean(), change_df["change"].mean())

    def _select_parents(self, candidates: list[Candidate], n_parents: int) -> list[Candidate]:
        """
        NSGA-II parent selection using fast non-dominated sort and crowding distance.
        Sets candidates' ranks and distance attributes.
        """
        fronts, ranks = nsga2_utils.fast_non_dominated_sort(candidates)
        for candidate, rank in zip(candidates, ranks):
            candidate.rank = rank
        parents = []
        for front in fronts:
            # Compute crowding distance here even though it's technically not necessary now
            # so that later we can sort by distance
            crowding_distance = nsga2_utils.calculate_crowding_distance(front)
            for candidate, distance in zip(front, crowding_distance):
                candidate.distance = distance
            if len(parents) + len(front) > n_parents:  # If adding this front exceeds num_parents
                front = sorted(front, key=lambda c: c.distance, reverse=True)
                parents += front[:n_parents - len(parents)]
                break
            parents += front
        return parents
    
    def _tournament_selection(self, sorted_parents: list[Candidate]) -> tuple[Candidate, Candidate]:
        """
        Takes two random parents and compares their indices since this is a measure of their performance.
        Note: It is possible for this function to select the same parent twice.
        """
        idx1 = min(random.choices(range(len(sorted_parents)), k=2))
        idx2 = min(random.choices(range(len(sorted_parents)), k=2))
        return sorted_parents[idx1], sorted_parents[idx2]

    def _make_new_pop(self, parents: list[Candidate], pop_size: int, gen:int) -> list[Candidate]:
        """
        Makes new population by creating children from parents.
        We use tournament selection to select parents for crossover.
        """
        sorted_parents = sorted(parents, key=lambda c: (c.rank, -c.distance))
        children = []
        for i in range(pop_size):
            parent1, parent2 = self._tournament_selection(sorted_parents)
            child = Candidate.from_crossover(parent1, parent2, self.p_mutation, gen, i)
            children.append(child)
        return children

    def neuroevolution(self, save_path: Path):
        """
        Main Neuroevolution Loop that performs NSGA-II.
        After initializing the first population randomly, goes through 3 steps in each generation:
        1. Evaluate candidates
        2. Select parents
        2a Log performance of parents
        3. Make new population from parents
        """
        if save_path.exists():
            shutil.rmtree(save_path)
        save_path.mkdir(parents=True, exist_ok=False)
        results = []
        parents = [Candidate(**self.candidate_params, gen=1, cand_id=i) for i in range(self.pop_size)]
        # Seeding the first generation with trained models
        if self.seed_dir:
            seed_paths = list(self.seed_dir.glob("*.pt"))
            for i, seed_path in enumerate(seed_paths):
                print(f"Seeding with {seed_path}...")
                parents[i].load_state_dict(torch.load(seed_path))

        offspring = []
        for gen in tqdm(range(1, self.n_generations+1)):
            # Set up candidates by merging parent and offspring populations
            candidates = parents + offspring
            self._evaluate_candidates(candidates)

            # NSGA-II parent selection
            parents = self._select_parents(candidates, self.pop_size)

            # Record the performance of the most successful candidates
            results.append(self._record_candidate_avgs(gen+1, parents))
            self._record_gen_results(gen, parents, save_path)

            # If we aren't on the last generation, make a new population
            if gen < self.n_generations:
                offspring = self._make_new_pop(parents, self.pop_size, gen)

        results_df = pd.DataFrame(results)
        results_df.to_csv(save_path / "results.csv", index=False)

        return parents
    
    def _record_gen_results(self, gen: int, candidates: list[Candidate], save_path: Path) -> None:
        """
        Records the state of all the candidates.
        Save the pareto front to disk.
        """
        # Save statistics of candidates
        gen_results = [c.record_state() for c in candidates]
        gen_results_df = pd.DataFrame(gen_results)
        gen_results_df.to_csv(save_path / f"{gen}.csv", index=False)

        # Save rank 1 candidate state dicts
        (save_path / f"{gen}").mkdir(parents=True, exist_ok=True)
        pareto_candidates = [c for c in candidates if c.rank == 1]
        for c in pareto_candidates:
            torch.save(c.state_dict(), save_path / f"{gen}" / f"{c.gen}_{c.cand_id}.pt")

    def _record_candidate_avgs(self, gen: int, candidates: list[Candidate]) -> dict:
        """
        Gets the average eluc and change for a population of candidates.
        """
        avg_eluc = np.mean([c.metrics[0] for c in candidates])
        avg_change = np.mean([c.metrics[1] for c in candidates])
        return {"gen": gen, "eluc": avg_eluc, "change": avg_change}     

    def prescribe_land_use(self, context_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Wrapper for prescribe method that loads a candidate from disk using an id.
        Valid kwargs:
            cand_id: str, the ID of the candidate to load
            results_dir: Path, the directory where the candidate is stored
        Then takes in a context dataframe and prescribes actions.
        """
        candidate = Candidate(**self.candidate_params)
        gen = int(kwargs["cand_id"].split("_")[0])
        state_dict = torch.load(kwargs["results_dir"] / f"{gen + 1}" / f"{kwargs['cand_id']}.pt")
        candidate.load_state_dict(state_dict)
        
        context_actions_df = self._prescribe(candidate, context_df)
        return context_actions_df
    