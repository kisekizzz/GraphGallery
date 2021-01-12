#!/usr/bin/env python
# coding: utf-8

# In[53]:


from graphgallery.data.graph import HomoGraph
import numpy as np
import scipy.sparse as sp


def test_HomoGraph():
    adj1 = sp.csr_matrix((np.array([2,1,2,3,1]), np.array([0,1,2,3,0]), np.array([0,2,4,4,5,5])), shape=(5, 5))
    
#     adj2 =  np.array([[0, 1, 0, 0, 0],
#                        [0, 0, 1, 2, 0],
#                        [2, 0, 0, 0, 0],
#                        [1, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0]])

#    adj = np.array([adj1,adj2])
    n_attr1 = np.array([[1,2],[0,3],[1,2],[4,5],[7,8]])
#     n_attr2 = np.array([[0,2],[0,2],[2,3],[6,0],[2,8]])
    n_attr = np.array([n_attr1, n_attr2])
    n_l = np.array([0,1,2,3,4])
    

    myGraph = HomoGraph(adj_matrix=adj1, node_attr=n_attr1, node_label=n_l)
    
    
    assert myGraph.num_nodes == 5
    assert myGraph.num_edges == 5 
    assert myGraph.num_graphs == 1
    assert myGraph.num_node_attrs == 2
    assert myGraph.num_node_classes == 6
    assert myGraph.num_attrs == 2
    assert myGraph.num_classes == 6
    assert (myGraph.A.toarray() == adj1.toarray()).all()
    assert (myGraph.y == n_l).all()
    D = myGraph.d
    assert (D[0] == np.array([3,1,2,3,0],dtype=np.float32)).all()
    assert (D[1] == np.array([3,5,0,1,0],dtype=np.float32)).all()
    assert myGraph.is_directed() == True
    assert myGraph.is_singleton() == True
    assert myGraph.is_selfloops() == True
    assert myGraph.is_weighted() == True
    assert myGraph.is_binary() == False
test_HomoGraph()
    
    


# In[ ]:




