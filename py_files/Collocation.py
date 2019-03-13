import nltk
import operator
import numpy as np

class Collocation:

    # Initialize collocation with tuple of words
    def __init__(self, words):
        self.words = words
        self.offsets = []
        self.frequency = 0

    # Add a new occurrence with given offset
    def add(self, offset):
        self.frequency += 1
        self.offsets.append(offset)

    # Get the mean offset
    def mean(self):
        return np.mean(np.array(self.offsets))

    # Get the standard deviation of the offsets
    def standard_deviation(self):
        return np.std(np.array(self.offsets))

    # Print collocation
    def __repr__(self):
        return f"{self.words} {self.frequency} {self.mean()} {self.standard_deviation()}"
