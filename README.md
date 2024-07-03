# Hypergraph_Encodings

# Organisation of repo

```data/``` 

contains the coauthorship and cocitation data.

```src/```

contains functions for the package.

```scripts/``` 

contains the interface.


# Data

Presented as dictionaries. For coauthorship, Keys are authors, values are int (papers they participate in).

# How to run

```
conda create -n encodings_venv python=3.11
conda activate encodings_venv
pip install -e .
```

# Tests

```
pytest
``` 
from root.

For a file in particular:


```
pytest tests/test_curvature.py
``` 


# Julia

A bit hacky for now.

Give permission:

chmod +x /Users/pellegrinraphael/Desktop/Repos_GNN/Hypergraph_Encodings/src/orchid/orchid_interface.jl

TODO:
using Pkg
Pkg.add("Documenter")
julia -e 'using Documenter; Documenter.generate("docs")'

To generate html docs.