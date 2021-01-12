#!/usr/bin/env python
# coding: utf-8

# In[19]:


import unittest
from graphgallery.data.graph import Graph
import numpy as np
import scipy.sparse as sp



# In[20]:


def test_graph():
    adj = sp.csr_matrix((np.array([2,1,2,3,1]), np.array([0,1,2,3,0]), np.array([0,2,4,4,5,5])), shape=(5, 5))
    
    graph = Graph(adj_matrix=adj, node_label=np.array([1,1,3,4,5]))

    assert all(graph.neighbors(0) == [0,1])
    assert all(graph.neighbors(1) == [2,3])
    assert all(graph.neighbors(2) == [0])
    assert all(graph.neighbors(3) == [])
    
    unw_graph = graph.to_unweighted()
    unw_adj = unw_graph.adj_matrix
    assert unw_adj.toarray().tolist() == [[1, 1, 0, 0, 0], 
                                          [0, 0, 1, 1, 0], 
                                          [0, 0, 0, 0, 0], 
                                          [1, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0]]
    
    und_graph = graph.to_unweighted().to_undirected()
    und_adj = und_graph.adj_matrix
    assert und_adj.toarray().tolist() == [[1, 1, 0, 1, 0], 
                                          [1, 0, 1, 1, 0], 
                                          [0, 1, 0, 0, 0], 
                                          [1, 1, 0, 0, 0], 
                                          [0, 0, 0, 0, 0]]
    assert graph.eliminate_selfloops().adj_matrix.toarray().tolist() == [[0, 1, 0, 0, 0], 
                                                                         [0, 0, 2, 3, 0], 
                                                                         [0, 0, 0, 0, 0], 
                                                                         [1, 0, 0, 0, 0],
                                                                         [0, 0, 0, 0, 0]]

    
    assert graph.eliminate_classes(threshold=2).adj_matrix.toarray().tolist() == []
    assert graph.eliminate_classes(threshold=1).adj_matrix.toarray().tolist() == [[2,1],
                                                                                  [0,0]]
    
    assert graph.add_selfloops().adj_matrix.toarray().tolist() == [[1, 1, 0, 0, 0],
                                                                   [0, 1, 2, 3, 0],
                                                                   [0, 0, 1, 0, 0],
                                                                   [1, 0, 0, 1, 0],
                                                                   [0, 0, 0, 0, 1]]
    
    assert graph.standardize().adj_matrix.toarray().tolist() == [[0, 1, 0, 1],
                                                                 [1, 0, 1, 1],
                                                                 [0, 1, 0, 0],
                                                                 [1, 1, 0, 0]]
    
    assert list(graph.nxgraph().nodes(data=True)) == [(0, {}), (1, {}), (2, {}), (3, {}), (4, {})]
    assert list(graph.nxgraph().edges(data=True)) == [(0, 0, {'weight': 2.0}), (0, 1, {'weight': 1.0}), (1, 2, {'weight': 2.0}), (1, 3, {'weight': 3.0}), (3, 0, {'weight': 1.0})]
    
    assert graph.subgraph(nodes_to_remove=[]).adj_matrix.toarray().tolist() == [[2, 1, 0, 0, 0],
                                                                                [0, 0, 2, 3, 0],
                                                                                [0, 0, 0, 0, 0],
                                                                                [1, 0, 0, 0, 0],
                                                                                [0, 0, 0, 0, 0]]
    assert graph.subgraph(nodes_to_remove=[2,3]).adj_matrix.toarray().tolist() == [[2, 1, 0],
                                                                                   [0, 0, 0],
                                                                                   [0, 0, 0]]
    assert graph.subgraph(nodes_to_remove=[0,1,2,3,4]).adj_matrix.toarray().tolist() == []
    
    assert graph.subgraph(nodes_to_keep=[0,1,4]).adj_matrix.toarray().tolist() == [[2, 1, 0],
                                                                                   [0, 0, 0],
                                                                                   [0, 0, 0]]
    assert graph.subgraph(nodes_to_keep=[0,1,2,3,4]).adj_matrix.toarray().tolist() == [[2, 1, 0, 0, 0],
                                                                                       [0, 0, 2, 3, 0],
                                                                                       [0, 0, 0, 0, 0],
                                                                                       [1, 0, 0, 0, 0],
                                                                                       [0, 0, 0, 0, 0]]

    
    
    
    
    

    
    


# In[ ]:





# ##### text_graph()

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




