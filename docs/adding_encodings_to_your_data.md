# Compute Encodings Module

This module provides classes and utilities for computing and saving various 
hypergraph encodings. It supports both local and cluster environments with 
different data handling strategies.

## Overview

The `compute_encodings` module contains classes that process hypergraphs 
and compute multiple types of encodings including:
- Random Walk encodings (EE, EN, WE)
- Laplacian encodings (Hodge, Normalized)
- Curvature encodings (ORC, FRC)
- Degree encodings (LDP)
(see src/encodings_hnns for the backends)

## Data Location Strategy

### Local Environment
- **Data Directory**: 
- `data/hypergraph_classification_datasets/` 
- `data/graph_classification_datasets/`
- `data/coauthorship/`
- `data/cocitation/`



#### Hypergraphs

`cocitation`(citeseere, cora, pubmed) and `coauthorship` (cora, dblp):
the data already already come as hypergraphs. Task is node classification.

#### Graphs

The naming is not the best:

`hypergraph_classification_datasets` (collab, imdb, reddit, enzymes, mutag,, proteins):
graph that we can turn into hypergraphy by clique expansion.
Task is graph/hypergraph level classification.

`graph_classification_datasets`:
peptide_struct and peptide_func from LRGB dataset. Graph-level classification
or regression task.

#### Computing the encodings

When computing the encodings on the data:
- **Individual Files**: Stored in `data/*/individual_files/` for granular access
(ie hypergraph by hypergraph)
- **Combined Files**: Stored in `data/*` by combining the individual hypergraph
file.

Eg

```
(encodings_venv) ➜  hypergraph_classification_datasets git:(ruff_cleanup) ✗ ls -lt
total 2971896
-rw-r--r--      1 pellegrinraphael  staff   62178476 10 Jul 13:10 reddit_hypergraphs_with_encodings_ldp.pickle
-rw-r--r--      1 pellegrinraphael  staff   55368998 10 Jul 13:09 reddit_hypergraphs_with_encodings_frc.pickle
-rw-r--r--      1 pellegrinraphael  staff   55368998 10 Jul 13:09 reddit_hypergraphs_with_encodings_orc.pickle
-rw-r--r--      1 pellegrinraphael  staff  157483804 10 Jul 13:09 reddit_hypergraphs_with_encodings_lape_normalized.pickle
-rw-r--r--      1 pellegrinraphael  staff  157483804 10 Jul 13:09 reddit_hypergraphs_with_encodings_lape_hodge.pickle
-rw-r--r--      1 pellegrinraphael  staff  157509704 10 Jul 13:09 reddit_hypergraphs_with_encodings_rw_WE.pickle
-rw-r--r--      1 pellegrinraphael  staff  157509704 10 Jul 13:09 reddit_hypergraphs_with_encodings_rw_EN.pickle
-rw-r--r--      1 pellegrinraphael  staff  157509704 10 Jul 13:09 reddit_hypergraphs_with_encodings_rw_EE.pickle
drwxr-xr-x  65535 pellegrinraphael  staff    2534720 10 Jul 13:09 individual_files
(encodings_venv) ➜  hypergraph_classification_datasets git:(ruff_cleanup) ✗ cd individual_files 
(encodings_venv) ➜  individual_files git:(ruff_cleanup) ✗ ls -lt
total 3237184
-rw-r--r--  1 pellegrinraphael  staff  225000 10 Jul 13:09 reddit_hypergraphs_with_encodings_ldp_count_1171.pickle
-rw-r--r--  1 pellegrinraphael  staff  200424 10 Jul 13:09 reddit_hypergraphs_with_encodings_frc_count_1171.pickle
-rw-r--r--  1 pellegrinraphael  staff  200424 10 Jul 13:09 reddit_hypergraphs_with_encodings_orc_count_1171.pickle
-rw-r--r--  1 pellegrinraphael  staff  569064 10 Jul 13:09 reddit_hypergraphs_with_encodings_lape_normalized_count_1171.pickle
-rw-r--r--  1 pellegrinraphael  staff  569064 10 Jul 13:09 reddit_hypergraphs_with_encodings_lape_hodge_count_1171.pickle
-rw-r--r--  1 pellegrinraphael  staff  569064 10 Jul 13:09 reddit_hypergraphs_with_encodings_rw_WE_count_1171.pickle
```

### Cluster Environment
Some of the data was saved on the Harvard cluster for us, namely the LRGB data
- **LRGB Datasets**: `/n/holyscratch01/mweber_lab/lrgb_datasets/`

## Core Classes

### `EncodingsSaverBase`
Base class providing common functionality for encoding computation and storage.

**Key Features:**
- Multiprocessing support for parallel hypergraph processing (processes all hypergraph from one file in parallel)
- Automatic directory creation and file management
- Error handling for disconnected components
- Support for multiple encoding types

**Methods:**
- `_process_hypergraph()`: Processes individual hypergraphs
- `_process_file()`: Processes entire files with parallel execution. A file would contain multiple graphs/hypergraphs.

### `EncodingsSaverForCC/CA`

Need to redo: dor CC/CA (Cocitation /Coauthorship)
Need to recompute them and re-find them from the cluster.
Need to add the class to do that.

### `EncodingsSaver`
Specialized class for hypergraph classification datasets (Proteins, Enzymes, Mutag, IMDB, Collab, Reddit).

**Supported Datasets:**
- `proteins_hypergraphs`
- `enzymes_hypergraphs`
- `mutag_hypergraphs`
- `imdb_hypergraphs`
- `collab_hypergraphs`
- `reddit_hypergraphs`

Add your owns!

### `EncodingsSaverLRGB`
Specialized class for Long Range Graph Benchmark (LRGB) datasets.

**Supported Datasets:**
- `peptidesstruct` (Chemistry - Graph Regression)
- `peptidesfunc` (Chemistry - Graph Classification)

Add your owns!

**Features:**
- Converts PyTorch Geometric graphs to hypergraph format
- Handles train/val/test splits
- Processes edge features and node features

## Encoding Types

### Random Walk Encodings
- **EE**: Edge-Edge random walks
- **EN**: Edge-Node random walks  
- **WE**: Weighted Edge random walks

### Laplacian Encodings
- **Hodge**: Hodge Laplacian
- **Normalized**: Normalized Laplacian

### Curvature Encodings
- **ORC**: Ollivier-Ricci curvature
- **FRC**: Forman-Ricci curvature

### Degree Encodings
- **LDP**: Local Degree Profile



## Usage Examples

Call the script:

```
python scripts/compute_encodings/compute_and_save_encodings.py
```

### Local Processing
```python
from compute_encodings.encoding_saver_hgraphs import EncodingsSaver

# Process hypergraph datasets locally
saver = EncodingsSaver("hypergraph_classification_datasets")
results = saver.compute_encodings()
```

### Cluster Processing (LRGB)
```python
from compute_encodings.encoding_saver_lrgb import EncodingsSaverLRGB

# Process LRGB datasets on cluster
saver = EncodingsSaverLRGB("graph_classification_datasets")
converted_data = saver.load_and_convert_lrgb_datasets(
    "/n/holyscratch01/mweber_lab/lrgb_datasets", 
    "peptidesstruct"
)
```

## Error Handling

The module handles several types of errors:
- **DisconnectedError**: When hypergraphs have disconnected components
- **FileNotFoundError**: When data files are missing
- **MemoryError**: For large datasets on limited resources

## Performance Considerations

### Local Environment
- Use multiprocessing for parallel hypergraph processing
- Monitor memory usage for large datasets

### Cluster Environment
- Utilize cluster storage for large LRGB datasets
- Use batch processing for multiple datasets
- Monitor job queue and resource allocation

## Notes

- Individual files are saved for granular access and debugging
- Combined files are created for batch processing and training
- The module automatically creates necessary directories


# Eg my local folder is

STILL NEED TO EXPLAIN THE DIFFERENT STRUCUTRE FOR 
CA/CC: features, hypegraphs, labels. What are splits? +
RUN THE ENCODINGS ON THESE/PROVIDE THE CLASS FOR THESE.

```
.
├── BREC_Data
│   ├── 4vtx.npy
│   ├── basic.npy
│   ├── cfi.npy
│   ├── dr.npy
│   ├── extension.npy
│   ├── notes.txt
│   ├── regular.npy
│   └── str.npy
├── Rook_Shrikhande
│   ├── rook_graph.g6
│   └── shrikhande.g6
├── coauthorship
│   ├── cora
│   │   ├── features.pickle
│   │   ├── hypergraph.pickle
│   │   ├── labels.pickle
│   │   └── splits
│   │       ├── 1.pickle
│   │       ├── 10.pickle
│   │       ├── 2.pickle
│   │       ├── 3.pickle
│   │       ├── 4.pickle
│   │       ├── 5.pickle
│   │       ├── 6.pickle
│   │       ├── 7.pickle
│   │       ├── 8.pickle
│   │       └── 9.pickle
│   └── dblp
│       ├── features.pickle
│       ├── hypergraph.pickle
│       ├── labels.pickle
│       └── splits
│           ├── 1.pickle
│           ├── 10.pickle
│           ├── 2.pickle
│           ├── 3.pickle
│           ├── 4.pickle
│           ├── 5.pickle
│           ├── 6.pickle
│           ├── 7.pickle
│           ├── 8.pickle
│           └── 9.pickle
├── cocitation
│   ├── citeseer
│   │   ├── features.pickle
│   │   ├── hypergraph.pickle
│   │   ├── labels.pickle
│   │   └── splits
│   │       ├── 1.pickle
│   │       ├── 10.pickle
│   │       ├── 2.pickle
│   │       ├── 3.pickle
│   │       ├── 4.pickle
│   │       ├── 5.pickle
│   │       ├── 6.pickle
│   │       ├── 7.pickle
│   │       ├── 8.pickle
│   │       └── 9.pickle
│   ├── cora
│   │   ├── features.pickle
│   │   ├── hypergraph.pickle
│   │   ├── labels.pickle
│   │   └── splits
│   │       ├── 1.pickle
│   │       ├── 10.pickle
│   │       ├── 2.pickle
│   │       ├── 3.pickle
│   │       ├── 4.pickle
│   │       ├── 5.pickle
│   │       ├── 6.pickle
│   │       ├── 7.pickle
│   │       ├── 8.pickle
│   │       └── 9.pickle
│   └── pubmed
│       ├── features.pickle
│       ├── hypergraph.pickle
│       ├── labels.pickle
│       └── splits
│           ├── 1.pickle
│           ├── 10.pickle
│           ├── 2.pickle
│           ├── 3.pickle
│           ├── 4.pickle
│           ├── 5.pickle
│           ├── 6.pickle
│           ├── 7.pickle
│           ├── 8.pickle
│           └── 9.pickle
├── graph_classification_datasets
│   ├── individual_files
│   │   ├── peptidesstruct_with_encodings_frc_count_0.pickle
│   │   ├── peptidesstruct_with_encodings_frc_count_1.pickle
        ...

│   │   └── peptidesstruct_with_encodings_rw_WE_count_9999.pickle
│   ├── peptidesstruct_all_with_encodings_frc.pickle
│   ├── peptidesstruct_all_with_encodings_lape_hodge.pickle
│   ├── peptidesstruct_all_with_encodings_lape_normalized.pickle
│   ├── peptidesstruct_all_with_encodings_ldp.pickle
│   ├── peptidesstruct_all_with_encodings_orc.pickle
│   ├── peptidesstruct_hypergraphs.pickle
│   ├── peptidesstruct_hypergraphs_test.pickle
│   ├── peptidesstruct_hypergraphs_train.pickle
│   ├── peptidesstruct_hypergraphs_val.pickle
│   ├── peptidesstruct_test_with_encodings_frc.pickle
│   ├── peptidesstruct_test_with_encodings_lape_hodge.pickle
│   ├── peptidesstruct_test_with_encodings_lape_normalized.pickle
│   ├── peptidesstruct_test_with_encodings_ldp.pickle
│   ├── peptidesstruct_test_with_encodings_orc.pickle
│   ├── peptidesstruct_train_with_encodings_frc.pickle
│   ├── peptidesstruct_train_with_encodings_lape_hodge.pickle
│   ├── peptidesstruct_train_with_encodings_lape_normalized.pickle
│   ├── peptidesstruct_train_with_encodings_ldp.pickle
│   ├── peptidesstruct_train_with_encodings_orc.pickle
│   ├── peptidesstruct_val_with_encodings_frc.pickle
│   ├── peptidesstruct_val_with_encodings_lape_hodge.pickle
│   ├── peptidesstruct_val_with_encodings_lape_normalized.pickle
│   ├── peptidesstruct_val_with_encodings_ldp.pickle
│   ├── peptidesstruct_val_with_encodings_orc.pickle
├── hypergraph_classification_datasets
│   ├── collab_hypergraphs.pickle
│   ├── collab_hypergraphs_with_encodings_frc.pickle
│   ├── collab_hypergraphs_with_encodings_orc.pickle
│   ├── enzymes_hypergraphs.pickle
│   ├── enzymes_hypergraphs_with_encodings_frc.pickle
│   ├── enzymes_hypergraphs_with_encodings_orc.pickle
│   ├── imdb_hypergraphs.pickle
│   ├── imdb_hypergraphs_with_encodings_frc.pickle
│   ├── imdb_hypergraphs_with_encodings_orc.pickle
│   ├── individual_files
│   │   ├── enzymes_hypergraphs_with_encodings_frc_count_0.pickle
│   │   ├── enzymes_hypergraphs_with_encodings_frc_count_1.pickle
        ...
│   │   └── proteins_hypergraphs_with_encodings_rw_WE_count_999.pickle
│   ├── mutag_hypergraphs.pickle
│   ├── mutag_hypergraphs_with_encodings_frc.pickle
│   ├── mutag_hypergraphs_with_encodings_orc.pickle
│   ├── note.txt
│   ├── proteins_hypergraphs.pickle
│   ├── proteins_hypergraphs_with_encodings_frc.pickle
│   ├── proteins_hypergraphs_with_encodings_lape_hodge.pickle
│   ├── proteins_hypergraphs_with_encodings_lape_normalized.pickle
│   ├── proteins_hypergraphs_with_encodings_ldp.pickle
│   ├── proteins_hypergraphs_with_encodings_orc.pickle
│   ├── proteins_hypergraphs_with_encodings_rw_EE.pickle
│   ├── proteins_hypergraphs_with_encodings_rw_EN.pickle
│   ├── proteins_hypergraphs_with_encodings_rw_WE.pickle
│   ├── reddit_hypergraphs.pickle
│   ├── reddit_hypergraphs_with_encodings_frc.pickle
│   └── reddit_hypergraphs_with_encodings_orc.pickle
├── notes.txt
├── peptidesstruct
│   ├── test.pt
│   ├── train.pt
│   └── val.pt
└── raw
    └── brec_v3.npy
```
