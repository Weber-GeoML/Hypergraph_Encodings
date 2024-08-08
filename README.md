# Hypergraph_Encodings

# Organisation of repo

```data/``` 

contains the coauthorship and cocitation data.

```src/```

contains functions for the package.

```scripts/``` 

contains the interface.


We follow the interface from UniGCN:

```
python scripts/train.py --data=coauthorship --dataset=dblp --model-name=UniSAGE 
```

```
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           data name (coauthorship/cocitation) (default:
                        coauthorship)
  --dataset DATASET     dataset name (e.g.: cora/dblp for coauthorship,
                        cora/citeseer/pubmed for cocitation) (default: cora)
  --model-name MODEL_NAME
                        UniGNN Model(UniGCN, UniGAT, UniGIN, UniSAGE...)
                        (default: UniSAGE)
  --first-aggregate FIRST_AGGREGATE
                        aggregation for hyperedge h_e: max, sum, mean
                        (default: mean)
  --second-aggregate SECOND_AGGREGATE
                        aggregation for node x_i: max, sum, mean (default:
                        sum)
  --add-self-loop       add-self-loop to hypergraph (default: False)
  --use-norm            use norm in the final layer (default: False)
  --activation ACTIVATION
                        activation layer between UniConvs (default: relu)
  --nlayer NLAYER       number of hidden layers (default: 2)
  --nhid NHID           number of hidden features, note that actually it's
                        #nhid x #nhead (default: 8)
  --nhead NHEAD         number of conv heads (default: 8)
  --dropout DROPOUT     dropout probability after UniConv layer (default: 0.6)
  --input-drop INPUT_DROP
                        dropout probability for input layer (default: 0.6)
  --attn-drop ATTN_DROP
                        dropout probability for attentions in UniGATConv
                        (default: 0.6)
  --lr LR               learning rate (default: 0.01)
  --wd WD               weight decay (default: 0.0005)
  --epochs EPOCHS       number of epochs to train (default: 200)
  --n-runs N_RUNS       number of runs for repeated experiments (default: 10)
  --gpu GPU             gpu id to use (default: 0)
  --seed SEED           seed for randomness (default: 1)
  --patience PATIENCE   early stop after specific epochs (default: 200)
  --nostdout            do not output logging to terminal (default: False)
  --split SPLIT         choose which train/test split to use (default: 1)
  --out-dir OUT_DIR     output dir (default: runs/test)
```


# Data

Presented as dictionaries. For coauthorship, Keys are authors, values are int (papers they participate in).

# How to run

Create a virtual env, activate it and install the required packages.

```
conda create -n encodings_venv python=3.11
conda activate encodings_venv
pip install -e .
```
You might need to install julia separately as well as the ORC routine calls julia code.

# Tests

We have a test suite that runs using pytest.

Simply run:

```
pytest
``` 
from root. That sould look for all tests and run them. You can run

```
pytest --verbose
```
for more prints.

For a file in particular, run:

```
pytest tests/test_curvature.py
``` 


# Julia

A bit hacky for now.

You might need to give permission to the file. Eg

```
chmod +x /Users/pellegrinraphael/Desktop/Repos_GNN/Hypergraph_Encodings/src/orchid/orchid_interface.jl
```




TODO:
using Pkg
Pkg.add("Documenter")
julia -e 'using Documenter; Documenter.generate("docs")'

To generate html docs.