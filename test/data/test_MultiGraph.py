#!/usr/bin/env python
# coding: utf-8

# In[1]:


import unittest
from graphgallery.data.multi_graph import MultiGraph
import numpy as np
import scipy.sparse as sp

def test_MultiGraph():

    adj1 = sp.csr_matrix((np.array([2,1,2,3,1]),
                         np.array([0,1,2,3,0]),
                         np.array([0,2,4,4,5,5])),
                        shape=(5, 5))
    
    adj2 = sp.csr_matrix((np.array([2,2,3,3,1]),
                         np.array([1,2,3,0,0]),
                         np.array([0,1,3,4,5,5])),
                        shape=(5, 5))
    adj = np.array([adj1,adj2])

    
    n_attr1 = np.array([[1,2],[0,3],[1,2],[4,5],[7,8]])
    n_attr2 = np.array([[0,2],[0,2],[2,3],[6,0],[2,8]])
    n_attr = np.array([n_attr1, n_attr2])
    
    n_l = np.array([[0,1,2,3,4],[1,0,2,3,4]])
    
    mGraph = MultiGraph(adj_matrix=adj,
                       node_attr=n_attr,
                       node_label=n_l)

    
    G = mGraph.to_unweighted()
   
    assert G.adj_matrix[0].toarray().tolist() == [[1, 1, 0, 0, 0],
                                                 [0, 0, 1, 1, 0],
                                                 [0, 0, 0, 0, 0],
                                                 [1, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0]]

    assert G.adj_matrix[1].toarray().tolist() == [[0, 1, 0, 0, 0],
                                                 [0, 0, 1, 1, 0],
                                                 [1, 0, 0, 0, 0],
                                                 [1, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0]]
    
    G = G.to_undirected()
    assert G.adj_matrix[0].toarray().tolist() == [[1, 1, 0, 1, 0],
                                                   [1, 0, 1, 1, 0],
                                                   [0, 1, 0, 0, 0],
                                                   [1, 1, 0, 0, 0],
                                                   [0, 0, 0, 0, 0]]
    
    assert G.adj_matrix[1].toarray().tolist() == [[0, 1, 1, 1, 0],
                                                   [1, 0, 1, 1, 0],
                                                   [1, 1, 0, 0, 0],
                                                   [1, 1, 0, 0, 0],
                                                   [0, 0, 0, 0, 0]]
    
    G = mGraph.eliminate_selfloops()
    assert G.adj_matrix[0].toarray().tolist() == [[0, 1, 0, 1, 0],
                                                 [1, 0, 1, 1, 0],
                                                 [0, 1, 0, 0, 0],
                                                 [1, 1, 0, 0, 0],
                                                 [0, 0, 0, 0, 0]]
    
    assert G.adj_matrix[1].toarray().tolist() == [[0, 1, 1, 1, 0],
                                                 [1, 0, 1, 1, 0],
                                                 [1, 1, 0, 0, 0],
                                                 [1, 1, 0, 0, 0],
                                                 [0, 0, 0, 0, 0]]
    G = mGraph.add_selfloops()
    assert G.adj_matrix[0].toarray().tolist() == [[1, 1, 0, 1, 0],
                                                 [1, 1, 1, 1, 0],
                                                 [0, 1, 1, 0, 0],
                                                 [1, 1, 0, 1, 0],
                                                 [0, 0, 0, 0, 1]]
    
    assert G.adj_matrix[1].toarray().tolist() == [[1, 1, 1, 1, 0],
                                                  [1, 1, 1, 1, 0],
                                                  [1, 1, 1, 0, 0],
                                                  [1, 1, 0, 1, 0],
                                                  [0, 0, 0, 0, 1]]
    
    

    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




