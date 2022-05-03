import os.path as osp
import torch

DATA_DIR = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
DEVICE = torch.device('cpu')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_WORKERS = 4
BATCH_SIZE = 1000
USE_GRAPHICAL_LASSO = False
