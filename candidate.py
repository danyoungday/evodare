"""
Contains the Candidate class, which is a simple feed-forward neural network that keeps track
of its own metrics and logging information.
"""
import numpy as np
from bitarray import bitarray

class Candidate():
    def __init__(self, n_params: int):
        self.bitstring = bitarray(n_params)
        self.bitstring.setall(0)

    @classmethod
    def from_crossover(cls, parent1, parent2, p_mutation: float, gen: int, cand_id: int):
        """
        Crossover two parents to create a child.
        Take a random 50/50 choice of either parent's weights
        """
        # Warning: This is enormous
        child = cls(len(parent1.bitstring))

        mask_array = np.random.choice([True, False], size=len(child.bitstring), p=[0.5, 0.5])

        child.mutate(p_mutation)
        return child
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the simple nn
        """
        out = self.model(X)
        return out
    
    def mutate(self, p_mutation: float):
        """
        Randomly flips each bit with probability p_mutation
        """
        # Warning: This is enormous
        mutate_array = np.random.choice([True, False], size=len(self.bitstring), p=[p_mutation, 1 - p_mutation])

        mutate_bitarray = bitarray(mutate_array)
        self.bitstring ^= mutate_bitarray
