import os
import numpy as np
import networkx as nx
import os.path as osp
import scipy.sparse as sp

from typing import Union, Optional, List, Tuple, Any

from .hetegraph import HeteGraph


class EdgeGraph(HeteGraph):
	"""Attributed labeled heterogeneous graph stored in 
		Numpy array form."""
	multiple = False

	@property
	def A(self):
		"""had not been tested"""
		n = self.num_nodes()
		graph = {}
		for i in range(self.edge_index[0]):
			head = self.edge_index[0][i]
			tail = self.edge_index[1][i]
		if graph.__contains__(head):
			graph[head].append(tail)
		else:
			graph[head] = [tail]
		adj = nx.adjacency_matrix(nx.from_dict_of_lists(
				graph, create_using=nx.DiGraph()))
		return adj


	def to_Graph(self):
		"""had not been tested"""
		A = self.A
		return Graph(adj_matrix=A, 
					 node_attr=self.node_attr,
					 node_label=self.node_label, 
					 node_graph_label=self.node_graph_label,
					 graph_attr=self.graph_attr,
					 graph_label=self.graph_label,
					 metadata=self.metadata,
					 copy=self.copy)
