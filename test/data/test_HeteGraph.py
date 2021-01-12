#!/usr/bin/env python
# coding: utf-8

# In[6]:


from graphgallery.data.hetegraph import HeteGraph
import numpy as np
import scipy.sparse as sp


def test_HeteGraph():
    edge_index = np.array([[0,1,2,3], [2,0,1,2]],dtype=np.int64)
    edge_weight = np.array([0.1, 0.2, 0.3, 0.5],dtype=np.float32)
    edge_attr = np.array([[0.1, 0.2], [0.2, 0.3], [0.4, 0.3], [0.6, 0.2]],dtype=np.float32)
    edge_label = np.array([0,2,1,1])
    node_attr = np.array([[0.2, 0.5, 0.6], [0.1, 0.3, 0.6], [0.5, 0.5, 0.2], [0.2, 0.1, 0.2]])
    node_label = np.array([0,1,2,2])
    myGraph = HeteGraph(edge_index=edge_index,
                       edge_weight=edge_weight,
                       edge_attr=edge_attr,
                       edge_label=edge_label,
                       node_attr=node_attr,
                       node_label=node_label)
    
    assert myGraph.num_nodes == 4
    assert myGraph.num_edges == 4
    assert myGraph.num_graphs == 1
    assert myGraph.num_node_attrs == 3
    assert myGraph.num_node_classes == 3
    assert (myGraph.e == edge_index).all()
    assert (myGraph.w == edge_weight).all()
    assert (myGraph.x == edge_attr).all()
    assert (myGraph.y == edge_label).all()
    


# In[ ]:





# In[ ]:





# In[ ]:




