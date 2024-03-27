"""
Contains the Candidate class, which is a simple feed-forward neural network that keeps track
of its own metrics and logging information.
"""
import numpy as np
from bitarray import bitarray
from tqdm import tqdm
import psutil

import torch
from transformers import AutoModelForCausalLM

class Candidate():
    def __init__(self, layer: int, n_params: int, p: float, gen: int, cand_id: int):
        # P is drop rate
        self.layer = layer
        self.n_params = n_params
        self.p = p
        self.bitstring = None

        self.gen = gen
        self.cand_id = cand_id

    def random_init(self):
        """
        Randomly initialize the bitstring
        """
        self.bitstring = np.random.choice([False, True], size=self.n_params, p=[self.p, 1 - self.p])

    @classmethod
    def from_crossover(cls, parent1, parent2, p_mutation: float):
        """
        Crossover two parents to create a child.
        Take a random 50/50 choice of either parent's weights.
        Then mutate.
        """
        child = cls(parent1.layer, parent1.n_params, parent1.p)
        parent_mask = np.random.choice([False, True], size=parent1.n_params)
        child.bitstring = np.where(parent_mask, parent1.bitstring, parent2.bitstring)
        child.mutate(p_mutation)
        return child

    def modify_model(self, base_layer, model_a):
        """
        Modify an LLM from the bitstring.
        Returns the old parameters
        """
        print(f"old equality: {model_a.model.layers[self.layer].parameters() == base_layer.parameters()}")
        old_weights = []
        param_count = 0
        for base_param, a_param in zip(base_layer.parameters(), model_a.model.layers[self.layer].parameters()):
            old_weights.append(a_param.data.clone())
            mask = self.bitstring[param_count:param_count + base_param.numel()].reshape(base_param.shape)

            # Zero out with probability p
            a_param.data[~mask] = base_param.data[~mask]

            # Scale by 1/(1-p)
            a_param.data[mask] = (a_param.data[mask] - self.p * base_param.data[mask]) / (1 - self.p)

            param_count += base_param.numel()
        
        return old_weights

    def restore_model(self, model, old_weights):
        """
        Restore the model to the old weights
        """
        for param, old_weight in zip(model.model.layers[self.layer].parameters(), old_weights):
            param.data = old_weight

    def merge_model(self, base_layer, model_a, model_b):
        """
        Does DARE on models a and b using base layer
        Then replaces layer in merged model with merged layer
        Returns old merged weights
        """
        old_a = self.modify_model(base_layer, model_a)
        old_b = self.modify_model(base_layer, model_b)

        for a_param, b_param in zip(model_a.model.layers[self.layer].parameters(), model_b.model.layers[self.layer].parameters()):
            a_param.data = (a_param.data + b_param.data) / 2


    def forward(self):
        pass
    
    def mutate(self, p_mutation: float):
        """
        Randomly flips each bit with probability p_mutation
        """
        self.bitstring = np.where(np.random.rand(self.n_params) < p_mutation, ~self.bitstring, self.bitstring)
