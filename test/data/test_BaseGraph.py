#!/usr/bin/env python
# coding: utf-8

# In[110]:


from graphgallery.data.base_graph import BaseGraph
import numpy as np



class MybaseGraph(BaseGraph):
    def __init__(self, *args, **kwargs):
        super(MybaseGraph, self).__init__(args, kwargs)
        
    def num_graphs(self):
        return 10
        

def test_base_graph():
    
    n_attr=np.array([[0,1,2],[1,2,3]])
    e_attr=np.array([[0,2,3],[1,2,3]])
    g_attr=np.array([[1,2],[2,1]])
    n_l=np.array([2,3,4])
    g_l=np.array([1,2,3])
    
    myGraph = BaseGraph(node_attr=n_attr, edge_attr=e_attr, 
                        graph_attr=g_attr, node_label=n_l,
                        graph_label=g_l, text_awg=np.array([0]))
    
    assert myGraph.is_node_attributed() == True
    assert myGraph.is_edge_attributed() == True
    assert myGraph.is_graph_attributed() == True
    assert myGraph.is_node_labeled() == True
    assert myGraph.is_edge_labeled() == False
    assert myGraph.is_graph_labeled() == True
    assert hasattr(myGraph, 'text_awg') == True
    
    
    assert 'text_awg' in myGraph.keys()
    assert myGraph.__contains__('text_awg') == True
    assert len(myGraph.keys()) == 6
    
    myGraph.text_awg = None
    assert 'text_awg' not in myGraph.keys()
    assert myGraph.__contains__('text_awg') == False
    assert len(myGraph.keys()) == 5
    
    items = myGraph.items(apply_fn=None)
    assert len(items) == len(myGraph.keys())
    
    items = myGraph.to_dict()
    assert len(items) == 5
    assert (items["node_attr"] == n_attr).all()
    assert (items["edge_attr"] == e_attr).all()
    assert (items["graph_attr"] == g_attr).all()
    assert (items["node_label"] == n_l).all()
    assert (items["graph_label"] == g_l).all()
    
    g = myGraph.__call__("node_attr","edge_attr")
    assert (next(g)==n_attr).all(), (next(g)==e_attr).all() == (True,True)
    
    

    g = myGraph.copy(deepcopy=False)
    assert (g.keys()) == (myGraph.keys())
    g.node_attr = None
    assert len(g.keys()) == len(myGraph.keys()) - 1

    g = myGraph.copy(deepcopy=True)
    assert id([g]) == id([myGraph])
    
    myGraph = MybaseGraph()
    assert myGraph.__len__() == 10
    
    
    
   
    
        
        

    


# In[ ]:





# In[ ]:




