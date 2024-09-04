import datetime
import os
import shutil
import time
from random import sample
import pickle

import config
import numpy as np
import path
import torch
import torch.nn.functional as F
from torch.optim import optimizer

# load data
from encodings_hnns.data_handling import load

### configure logger
from uniGCN.logger import get_logger
from uniGCN.prepare import fetch_data, initialise, accuracy

# File originally taken from UniGCN repo

# Check if CUDA is available and move tensors to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_accs: list = []
best_val_accs: list = []
best_test_accs: list = []


args = config.parse()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

use_norm: str = "use-norm" if args.use_norm else "no-norm"
add_self_loop: str = "add-self-loop" if args.add_self_loop else "no-self-loop"

model_name: str = args.model_name
nlayer: int = args.nlayer
dirname = f"{datetime.datetime.now()}".replace(" ", "_").replace(":", ".")

data_split = [0.5, 0.25, 0.25]


# load the data from data\hypergraph_classification_datasets/imdb_hypergraphs.pickle
with open("data/hypergraph_classification_datasets/imdb_hypergraphs.pickle", "rb") as f:
    imdb = pickle.load(f)

with open("data/hypergraph_classification_datasets/collab_hypergraphs.pickle", "rb") as f:
    collab = pickle.load(f)

with open("data/hypergraph_classification_datasets/reddit_hypergraphs.pickle", "rb") as f:
    reddit = pickle.load(f)

datasets = {"imdb": imdb, "collab": collab, "reddit": reddit}


# split the data
def split_data(data, split):
    n = len(data)
    indices = np.random.permutation(n)
    train_idx = indices[: int(split[0] * n)]
    val_idx = indices[int(split[0] * n) : int((split[0] + split[1]) * n)]
    test_idx = indices[int((split[0] + split[1]) * n) :]
    return data[train_idx], data[val_idx], data[test_idx]


