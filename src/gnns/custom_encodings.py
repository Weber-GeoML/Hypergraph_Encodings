"""
custom_encodings.py

This file contains the following custom encodings:
    - shortest path relative positional encodings
    - Ollivier-Ricci curvature structural encodings
    - Ollivier-Ricci curvature shortest path relative positional encodings
"""
import copy

import torch
from torch import Tensor
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.transforms import BaseTransform
from torch_geometric.data.datapipes import functional_transform

from GraphRicciCurvature.OllivierRicci import OllivierRicci

from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.FormanRicci4 import FormanRicci4


@functional_transform('edge_curvature')
class EdgeCurvature(BaseTransform):
    """
    This class computes a discrete curvature for each edge and adds it to the edge attributes.
    """
    def __init__(self, attr_name = 'edge_curv'):
        self.attr_name = attr_name
        

    def compute_orc(self, data: Data) -> Data:
        graph = to_networkx(data)
        
        # compute ORC
        orc = OllivierRicci(graph, alpha=0, verbose="ERROR")
        orc.compute_ricci_curvature()
    
        # add the ORC to the edge attributes
        edge_curv = torch.tensor([orc.G.edges[edge]["ricciCurvature"]["rc_curvature"] for edge in graph.edges()])
        
        # check if the edge attributes already exist, and if so, append the ORC to the existing edge attributes
        if data.edge_attr is not None:
            data.edge_attr = torch.cat((data.edge_attr, edge_curv.view(-1, 1)), dim=-1)
        else:
            data.edge_attr = edge_curv.view(-1, 1)

        return data

    
@functional_transform('local_curvature_profile')
class LocalCurvatureProfile(BaseTransform):
    """
    This class computes the local curvature profile structural encoding for each node in a graph.
    """
    def __init__(self, attr_name = 'lcp_se'):
        self.attr_name = attr_name
        

    def compute_orc(self, data: Data) -> Data:
        graph = to_networkx(data)
        
        # compute ORC
        orc = OllivierRicci(graph, alpha=0, verbose="ERROR")
        orc.compute_ricci_curvature()
    
        # get the neighbors of each node
        neighbors = [list(graph.neighbors(node)) for node in graph.nodes()]
    
        # compute the min, max, mean, std, and median of the ORC for each node
        min_orc = []
        max_orc = []
        mean_orc = []
        std_orc = []
        median_orc = []

        for node in graph.nodes():
            if len(neighbors[node]) > 0:
                min_orc.append(min([orc.G.edges[node, neighbor]["ricciCurvature"]["rc_curvature"] for neighbor in neighbors[node]]))
                max_orc.append(max([orc.G.edges[node, neighbor]["ricciCurvature"]["rc_curvature"] for neighbor in neighbors[node]]))
                mean_orc.append(np.mean([orc.G.edges[node, neighbor]["ricciCurvature"]["rc_curvature"] for neighbor in neighbors[node]]))
                std_orc.append(np.std([orc.G.edges[node, neighbor]["ricciCurvature"]["rc_curvature"] for neighbor in neighbors[node]]))
                median_orc.append(np.median([orc.G.edges[node, neighbor]["ricciCurvature"]["rc_curvature"] for neighbor in neighbors[node]]))
            else:
                min_orc.append(0)
                max_orc.append(0)
                mean_orc.append(0)
                std_orc.append(0)
                median_orc.append(0)
                                                                      
        # create a torch.tensor of dimensions (num_nodes, 5) containing the min, max, mean, std, and median of the ORC for each node
        lcp_pe = torch.tensor([min_orc, max_orc, mean_orc, std_orc, median_orc]).T
        # lcp_pe = torch.tensor([mean_orc, median_orc, std_orc]).T

        # move lcps to the GPU if available
        # if torch.cuda.is_available():
            # lcp_pe = lcp_pe.cuda()
    
        # add the local degree profile positional encoding to the data object
        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat((data.x, lcp_pe), dim=-1)
        else:
            data.x = torch.cat(lcp_pe, dim=-1)

        return data
    

    def forward(self, data: Data) -> Data: # renamed this forward from compute_orc_approx
        graph = to_networkx(data)
            
        # get the neighbors of each node
        neighbors = [list(graph.neighbors(node)) for node in graph.nodes()]
    
        min_orc = []
        max_orc = []

        def compute_upper_bound(node_1, node_2):
            deg_node_1 = len(neighbors[node_1])
            deg_node_2 = len(neighbors[node_2])
            num_triangles = len([neighbor for neighbor in neighbors[node_1] if neighbor in neighbors[node_2]])
            return num_triangles / np.max([deg_node_1, deg_node_2])

        def compute_lower_bound(node_1, node_2):
            deg_node_1 = len(neighbors[node_1])
            deg_node_2 = len(neighbors[node_2])
            num_triangles = len([neighbor for neighbor in neighbors[node_1] if neighbor in neighbors[node_2]])
            return -np.max([0, 1 - 1/deg_node_1 - 1/deg_node_2 - num_triangles/np.min([deg_node_1, deg_node_2])]) - np.max([0, 1 - 1/deg_node_1 - 1/deg_node_2 - num_triangles/np.max([deg_node_1, deg_node_2])]) + num_triangles/np.max([deg_node_1, deg_node_2])

        for node in graph.nodes():
            if len(neighbors[node]) > 0:
                min_orc.append(min([compute_lower_bound(node, neighbor) for neighbor in neighbors[node]]))
                max_orc.append(max([compute_upper_bound(node, neighbor) for neighbor in neighbors[node]]))
            else:
                min_orc.append(0)
                max_orc.append(0)
                                                                      
        lcp_pe = torch.tensor([min_orc, max_orc]).T
    
        # add the local degree profile positional encoding to the data object
        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x

            # make sure the local curvature profile positional encoding is on the same device as the node features
            if data.x.device != lcp_pe.device:
                lcp_pe = lcp_pe.to(data.x.device)
            data.x = torch.cat((data.x, lcp_pe), dim=-1)
        else:
            data.x = torch.cat(lcp_pe, dim=-1)

        return data 
    

    def compute_afrc_3(self, data: Data) -> Data:
        graph = to_networkx(data)
        
        # compute ORC
        afrc_3 = FormanRicci(graph)
        afrc_3.compute_ricci_curvature()
    
        # get the neighbors of each node
        neighbors = [list(graph.neighbors(node)) for node in graph.nodes()]
    
        # compute the min, max, mean, std, and median of the ORC for each node
        min_afrc_3 = [min([afrc_3.G[node][neighbor]['AFRC'] for neighbor in neighbors[node]]) for node in graph.nodes()]
        max_afrc_3 = [max([afrc_3.G[node][neighbor]['AFRC'] for neighbor in neighbors[node]]) for node in graph.nodes()]
        mean_afrc_3 = [np.mean([afrc_3.G[node][neighbor]['AFRC'] for neighbor in neighbors[node]]) for node in graph.nodes()]
        std_afrc_3 = [np.std([afrc_3.G[node][neighbor]['AFRC'] for neighbor in neighbors[node]]) for node in graph.nodes()]
        median_afrc_3 = [np.median([afrc_3.G[node][neighbor]['AFRC'] for neighbor in neighbors[node]]) for node in graph.nodes()] 
                                                                      
        # create a torch.tensor of dimensions (num_nodes, 5) containing the min, max, mean, std, and median of the ORC for each node
        lcp_pe = torch.tensor([min_afrc_3, max_afrc_3, mean_afrc_3, std_afrc_3, median_afrc_3]).T
    
        # add the local degree profile positional encoding to the data object
        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat((data.x, lcp_pe), dim=-1)
        else:
            data.x = torch.cat(lcp_pe, dim=-1)

        return data
    

    def compute_afrc_4(self, data: Data) -> Data:
        graph = to_networkx(data)
        
        # compute ORC
        afrc_4 = FormanRicci4(graph)
        afrc_4.compute_afrc_4()
    
        # get the neighbors of each node
        neighbors = [list(graph.neighbors(node)) for node in graph.nodes()]
    
        # compute the min, max, mean, std, and median of the ORC for each node
        min_afrc_4 = [min([afrc_4.G[node][neighbor]['AFRC_4'] for neighbor in neighbors[node]]) for node in graph.nodes()]
        max_afrc_4 = [max([afrc_4.G[node][neighbor]['AFRC_4'] for neighbor in neighbors[node]]) for node in graph.nodes()]
        mean_afrc_4 = [np.mean([afrc_4.G[node][neighbor]['AFRC_4'] for neighbor in neighbors[node]]) for node in graph.nodes()]
        std_afrc_4 = [np.std([afrc_4.G[node][neighbor]['AFRC_4'] for neighbor in neighbors[node]]) for node in graph.nodes()]
        median_afrc_4 = [np.median([afrc_4.G[node][neighbor]['AFRC_4'] for neighbor in neighbors[node]]) for node in graph.nodes()] 
                                                                      
        # create a torch.tensor of dimensions (num_nodes, 5) containing the min, max, mean, std, and median of the ORC for each node
        lcp_pe = torch.tensor([min_afrc_4, max_afrc_4, mean_afrc_4, std_afrc_4, median_afrc_4]).T

        # add the local degree profile positional encoding to the data object
        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat((data.x, lcp_pe), dim=-1)
        else:
            data.x = torch.cat(lcp_pe, dim=-1)

        return data
    

    def compute_frc(self, data: Data) -> Data:
        graph = to_networkx(data)
            
        # get the neighbors of each node
        neighbors = [list(graph.neighbors(node)) for node in graph.nodes()]

        # compute the FRC of each edge (u, v) as 4 - deg(u) - deg(v)
        def frc(u, v):
            return 4 - len(neighbors[u]) - len(neighbors[v])
        
        # compute the min, max, mean, std, and median of the FRC for each node
        min_frc = [min([frc(node, neighbor) for neighbor in neighbors[node]]) for node in graph.nodes()]
        max_frc = [max([frc(node, neighbor) for neighbor in neighbors[node]]) for node in graph.nodes()]
        mean_frc = [np.mean([frc(node, neighbor) for neighbor in neighbors[node]]) for node in graph.nodes()]
        std_frc = [np.std([frc(node, neighbor) for neighbor in neighbors[node]]) for node in graph.nodes()]
        median_frc = [np.median([frc(node, neighbor) for neighbor in neighbors[node]]) for node in graph.nodes()]

        # create a torch.tensor of dimensions (num_nodes, 5) containing the min, max, mean, std, and median of the ORC for each node
        lcp_pe = torch.tensor([min_frc, max_frc, mean_frc, std_frc, median_frc]).T

        # add the local degree profile positional encoding to the data object
        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat((data.x, lcp_pe), dim=-1)
        else:
            data.x = torch.cat(lcp_pe, dim=-1)

        return data
    

@functional_transform('alt_local_curvature_profile')
class AltLocalCurvatureProfile(BaseTransform):
    """
    This class computes the local curvature profile structural encoding for each node in a graph.
    """
    def __init__(self, attr_name = 'alt_lcp_se'):
        self.attr_name = attr_name
        

    def compute_orc(self, data: Data) -> Data:
        graph = to_networkx(data)
        
        # compute ORC
        orc = OllivierRicci(graph, alpha=0, verbose="ERROR")
        orc.compute_ricci_curvature()
    
        # get the neighbors of each node
        neighbors = [list(graph.neighbors(node)) for node in graph.nodes()]
    
        # for each node, order the ORC of its neighbors in ascending order
        ordered_orc = []
        for node in graph.nodes():
            if len(neighbors[node]) > 0:
                ordered_orc.append(sorted([orc.G.edges[node, neighbor]["ricciCurvature"]["rc_curvature"] for neighbor in neighbors[node]]))
            else:
                ordered_orc.append([0])

        # crop the ordered ORC to the first 3 and the last 2 values, and pad with 0s if necessary
        cropped_orc = []
        for node in graph.nodes():
            if len(ordered_orc[node]) > 5:
                cropped_orc.append(ordered_orc[node][:3] + ordered_orc[node][-2:])
            elif len(ordered_orc[node]) < 5:
                cropped_orc.append(ordered_orc[node] + [0] * (5 - len(ordered_orc[node])))
            else:
                cropped_orc.append(ordered_orc[node])

        # create a torch.tensor of dimensions (num_nodes, 5) containing the ordered ORC for each node
        alt_lcp_pe = torch.tensor(cropped_orc)
    
        # add the local degree profile positional encoding to the data object
        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat((data.x, alt_lcp_pe), dim=-1)
        else:
            data.x = torch.cat(alt_lcp_pe, dim=-1)

        return data