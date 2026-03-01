import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # s set CPU seed.
    torch.cuda.manual_seed(seed)  # set current GPU seed.
    torch.cuda.manual_seed_all(seed)  # set all GPU seeds.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
