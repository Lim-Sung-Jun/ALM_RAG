import torch
import numpy as np
import random
from omegaconf import OmegaConf

def get_config(args):
    # 파일에서 config 가져오기에 좋다.
    config = OmegaConf.load(args.config)
    return config

# torch, numpy, random, cudnn
def fix_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 학습 속도가 느려질 수 있다.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
