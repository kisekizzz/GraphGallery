#!/usr/bin/env python
# coding: utf-8

# In[43]:


from graphgallery.data.listgraph import ListGraph
from graphgallery.data.graph import HomoGraph
from graphgallery.data.hetegraph import HeteGraph
import numpy as np
import scipy.sparse as sp

def Bequals(g1, g2):
    return np.array([(g1.adj_matrix.toarray() == g2.adj_matrix.toarray()).all(),
                   (g1.node_attr == g2.node_attr).all(),
                   (g1.node_label == g2.node_label).all()]).all()



def test_listGraph():
    adj1 = sp.csr_matrix((np.array([2,1,2,3,1]), np.array([0,1,2,3,0]), np.array([0,2,4,4,5,5])), shape=(5, 5))
    n_attr1 = np.array([[1,2],[0,3],[1,2],[4,5],[7,8]])
    n_attr2 = np.array([[0,2],[0,2],[2,3],[6,0],[2,8]])
    n_attr = np.array([n_attr1, n_attr2])
    n_l = np.array([0,1,2,3,4,5])

    BGraph = HomoGraph(adj_matrix=adj1, node_attr=n_attr1, node_label=n_l)
    
    e_i = np.array([[0,1,2,3], [2,0,1,2]],dtype=np.int64)
    e_w = np.array([0.1, 0.2, 0.3, 0.5],dtype=np.float32)
    e_attr = np.array([[0.1, 0.2], [0.2, 0.3], [0.4, 0.3], [0.6, 0.2]],dtype=np.float32)
    e_l = np.array([0,2,1,1])
    n_attr = np.array([[0.2, 0.5, 0.6], [0.1, 0.3, 0.6], [0.5, 0.5, 0.2], [0.2, 0.1, 0.2]])
    n_l = np.array([0,1,2,2])
    
    HGraph = HeteGraph(edge_index=e_i,
                       edge_weight=e_w,
                       edge_attr=e_attr,
                       edge_label=e_l,
                       node_attr=n_attr,
                       node_label=n_l)
    
    LGraph = ListGraph(BGraph,HGraph)
    
    
    assert len(LGraph.graphs) == 2
    assert Bequals(BGraph,LGraph.graph)
    
    item = LGraph.__getitem__(0)
    assert item == LGraph.graph
    
    item = LGraph.__getitem__(1)
    assert item == LGraph.graphs[1]
    
    g = LGraph.__iter__()
    item = next(g)
    assert item == LGraph.graph
    
    item = next(g)
    assert item == LGraph.graphs[1]
    
    assert LGraph.__len__() == 2


    




