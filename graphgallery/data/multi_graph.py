import os
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp

from functools import partial
from copy import copy as copy_fn

from typing import Union, Optional, List, Tuple, Any

from .homograph import HomoGraph
from .graph import Graph
from .apply import index_select
from ..data_type import is_intscalar
from .preprocess import create_subgraph


class MultiGraph(HomoGraph):
    """Multiple attributed labeled homogeneous graph stored in a list of 
        sparse matrices form."""
    multiple = True

    def to_undirected(self):
        """Convert to an undirected graph (make adjacency matrix symmetric)."""
        if self.is_weighted():
            raise ValueError(
                "Convert to unweighted graph first. Using 'graph.to_unweighted()'."
            )
        else:
            G = self.copy()
            for i, A in enumerate(G.adj_matrix):
                A = A.maximum(A.T)
                G.adj_matrix[i] = A
            return G
        

    def to_unweighted(self):
        """Convert to an unweighted graph (set all edge weights to 1)."""
        G = self.copy()
        for i, A in enumerate(G.adj_matrix):
            A = sp.csr_matrix(
                (np.ones_like(A.data), A.indices, A.indptr), shape=A.shape)
            G.adj_matrix[i] = A
        return G
        
        
    def eliminate_selfloops(self):
        """Remove self-loops from the adjacency matrix."""
        G = self.copy()
        for i, A in enumerate(G.adj_matrix):
            A = A - sp.diags(A.diagonal())
            A.eliminate_zeros()
            G.adj_matrix[i] = A
        return G

    def eliminate_classes(self, threshold=0):
        """Remove nodes from graph that correspond to a class of which there are less
        or equal than 'threshold'. Those classes would otherwise break the training procedure.
        """
        isChange = False
        G = None
        for i, n_l in enumerate(self.node_label):
            if n_l is None:
                continue
            # raise ValueError(
            # f"{self.node_label}")
            counts = np.bincount(n_l)
            nodes_to_remove = []
            removed = 0
            left = []
            
            for _class, count in enumerate(counts):
                if count <= threshold:
                    nodes_to_remove.extend(np.where(n_l == _class)[0])
                    removed += 1
                else:
                    left.append(_class)

            if removed > 0:
                if not isChange:
                    G = self.copy()
                    isChange = True
                G.adj_matrix[i] = self.subgraph(nodes_to_remove=nodes_to_remove).adj_matrix
                mapping = dict(zip(left, range(self.num_node_classes - removed)))
                G.node_label[i] = np.asarray(list(
                    map(lambda key: mapping[key], G.node_label)),
                    dtype=np.int32)
                continue
            
        if isChange:
            return G
        else:
            return self
            
    def eliminate_singleton(self):
        
        raise NotImplementedError

    def add_selfloops(self, value=1.0):
        """Set the diagonal."""
        G = self.eliminate_selfloops()
        for i, A in enumerate(G.adj_matrix):
            A = A + sp.diags(A.diagonal() + value)
            A.eliminate_zeros()
            G.adj_matrix[i] = A
        return G

    def standardize(self):
        """Select the largest connected components (LCC) of 
        the unweighted/undirected/no-self-loop graph."""
        raise NotImplementedError

    def nxgraph(self, directed: bool = True):
        """Get the network graph from adj_matrix."""
        raise NotImplementedErro

    def subgraph(self, *, nodes_to_remove=None, nodes_to_keep=None):
        return create_subgraph(self,
                               nodes_to_remove=nodes_to_remove,
                               nodes_to_keep=nodes_to_keep)

    def extra_repr(self):
        excluded = {"metadata", "mapping"}
        string = ""
        blank = ' ' * (len(self.__class__.__name__) + 1)
        for k, v in self.items():
            if v is None or k in excluded:
                continue
            string += f"{k}(num={len(v)}),\n{blank}"
        return string

    def __getitem__(self, index):
        try:
            apply_fn = partial(index_select, index=index)
            collects = self.dicts(apply_fn=apply_fn)
            if is_intscalar(index):
                # Single graph
                return Graph(**collects)
            else:
                G = self.copy()
                G.update(**collects)
                return G
        except IndexError as e:
            raise IndexError(f"Invalid index {index}.")
