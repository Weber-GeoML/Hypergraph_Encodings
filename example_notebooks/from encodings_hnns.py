import warnings

import hypernetx as hnx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from brec_analysis.analyse_brec_categories import analyze_brec_categories
from brec_analysis.compare_encodings_wrapper import compare_encodings_wrapper
from brec_analysis.plotting_graphs_and_hgraphs_for_brec import (
    plot_graph_pair, plot_hypergraph_pair)
from brec_analysis.utils_for_brec import (convert_nx_to_hypergraph_dict,
                                          nx_to_pyg)
from encodings_hnns.curvatures_frc import FormanRicci
from encodings_hnns.encodings import HypergraphEncodings
from encodings_hnns.expansions import plot_hypergraph_and_expansion
from encodings_hnns.laplacians import Laplacians
from encodings_hnns.liftings_and_expansions import lift_to_hypergraph

warnings.filterwarnings('ignore')
