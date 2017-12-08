import os
import random
import numpy as np
import torch

from common.training.options import gpu


class SimpleRandom():
    instance = None

    @staticmethod
    def get_seed():
        return os.getenv("RANDOM_SEED", 12459)

    @classmethod
    def set_seeds(cls):

        torch.manual_seed(SimpleRandom.get_seed())
        if gpu():
            torch.cuda.manual_seed_all(SimpleRandom.get_seed())
        np.random.seed(SimpleRandom.get_seed())
        random.seed(SimpleRandom.get_seed())