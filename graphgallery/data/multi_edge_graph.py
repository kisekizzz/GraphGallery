import os
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp

from collections import Counter
from copy import copy as copy_fn

from typing import Union, Optional, List, Tuple, Any
from .hetegraph import HeteGraph


class MultiEdgeGraph(HeteGraph):
	"""Multiple attributed labeled heterogeneous graph stored in 
		Numpy array form."""
	multiple = True
	



	@property
	def A(self):
		"""had not been tested"""
		graph_num = self.num_graphs()
		n = self.num_nodes()
		graph = {}
		adjs = []
		for e in self.edge_index:
			for i in range(e[0]):
				head = e[0][i]
				tail = e[1][i]
			if graph.__contains__(head):
				graph[head].append(tail)
			else:
				graph[head] = [tail]
			adj = nx.adjacency_matrix(nx.from_dict_of_lists(
					graph, create_using=nx.DiGraph()))
			adjs.append(adj)
		return np.array(adjs)
		
	def to_MultiGraph(self):
		"""had not been tested"""
		A = self.A
		return MultiGraph(adj_matrix=A, 
					 node_attr=self.node_attr,
					 node_label=self.node_label, 
					 node_graph_label=self.node_graph_label,
					 graph_attr=self.graph_attr,
					 graph_label=self.graph_label,
					 metadata=self.metadata,
					 copy=self.copy)