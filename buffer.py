import torch
import numpy as np
import random

class Memory(torch.utils.data.Dataset):
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self.input_samples = []
        self.output_samples = []

    def __getitem__(self, index):
        return self.input_samples[index], self.output_samples[index]
    def __len__(self):
        return self._max_memory

    def clear(self):
        self.input_samples = []
        self.output_samples = []
