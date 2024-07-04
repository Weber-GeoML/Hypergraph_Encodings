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