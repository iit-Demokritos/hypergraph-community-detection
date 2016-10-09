'''
Created on Mar 15, 2016

@author: ak
'''
import community
import itertools
import scipy as sp
import networkx as nx
from Utils.FormatHandlers import to_continuous_mapping, update_edge_list,\
    check_duplicate_indices

class CliqueGraph_nx(nx.Graph):
    """
    Class CliqueGraph_nx extending networkx Graph class
    """

    def __init__(self, edge_file,  node_tags=None):
        """
        Constructor, create simplified k-clique k-uniform hypergraph equivalent using networkx as base library
        
        Parameters
        ----------
        edge_file: list of lists
            list of lists containing hyperedge vertices,
        node_tags: list of strings
            list of node tags in order of appearance in hyperedge edge_file
            default=['P1', 'P2', 'P3', ...]
        
        """
        nx.Graph.__init__(self)
        self.mapping_id_to_index=None
        self.mapping_index_to_id=None
        if node_tags:
            self.node_tags=node_tags
        else:
            self.node_tags=['P'+str(i) for i in range(len(edge_file[0]))]
            
        CliqueGraph_nx.createCliqueGraph(self, edge_file, node_tags)
        
    @staticmethod
    def createCliqueGraph(g, edge_file, node_tags):
        """create simplified k-clique k-uniform hypergraph equivalent using networkx as base library
        
        Parameters
        ----------
        g: networkx graph
            used as base graph to append to
        edge_file: list of lists
            list of lists containing hyperedge vertices,
        node_tags: list of strings
            list of node tags in order of appearance in hyperedge edge_file
            default=['user', 'tag', resource']
        
        Return
        ------
        g: networkx undirected graph
        """
            
        if check_duplicate_indices(edge_file):
            edge_file, g.mapping_id_to_index, g.mapping_index_to_id, r=update_edge_list(edge_file)
            
        
        for h_edge in edge_file:
            attrs=[{'type':g.node_tags[i]} for i in range(len(h_edge))]
            g.add_nodes_from(zip(h_edge, attrs))
            g.add_edges_from(itertools.combinations(h_edge, 2))
        return g

            
    def modularity_maximization(self, partition=None):
        """Perform louvain's method for modularity maximization contained in 'community' library
        
        Parameters
        ----------
        partition: dict, optional
            the algorithm will start using this partition of the nodes. It's a dictionary where keys are their nodes and values the communities
            (doc taken from 'community' documentation http://perso.crans.org/aynaud/communities/api.html#community.best_partition)
            
        Returns
        -------
        com: list
            community array, entries are sorted by node index and grouped by node types/partites
        comms_d: dictionary
            dictionary containing {partiteName: { id: communityId , ...} , ... }
        node_tags: list of str
            order of partites, as found in hyperedges
        q: float
            modularity
        Raises
        ------
        NetworkXError: If the graph is not Eulerian.
            (doc taken from 'community' documentation http://perso.crans.org/aynaud/communities/api.html#community.best_partition)
        """
        #temporary class change for function to work
        self.__class__=nx.Graph
        d=community.best_partition(self, partition=partition)
        q=community.modularity(d, self)
        self.__class__=CliqueGraph_nx
        
        #community array
        com=[d[v] for v in sorted(d.keys())]
        
        #community dictionary
        comms_d={i:{} for i in self.node_tags}
        for n in self.nodes(data=True):
            if self.mapping_index_to_id:
                print(type(n))
                print(n)
                print(n[1]['type'])
                print(self.node_tags.index(n[1]['type']))
                comms_d[n[1]['type']][self.mapping_index_to_id[self.node_tags.index(n[1]['type'])][n[0]]]=d[n[0]]
            else:
                comms_d[n[1]['type']][n[0]]=d[n[0]]
        
        return com, comms_d, self.node_tags, q



#===============================================================================
# TEST - OK
#===============================================================================
if __name__=='__main__':
    edge_file=[[1,16,3,43],
               [1,4,5],
               [1,6,7,65],
               [2,8,9,76]]
    #===========================================================================
    # list_dicts=[{1:'user1',2:'user2'},
    #             {1:'tag1',4:'tag4',6:'tag6', 8:'tag8'},
    #             {3:'res3', 5:'res5', 7:'res7', 9:'res9'}]
    #===========================================================================
    node_tags=['user', 'tag', 'resource', 'what']
    
    #===========================================================================
    # g=CliqueGraph_nx()
    # g=CliqueGraph_nx.createCliqueGraph(g,edge_file, list_dicts)
    # print(len(g.nodes()))
    # print(g.nodes(data=True))
    # print(g.edges())
    # d=g.modularity_maximization()
    # 
    # r0,r1=CliqueGraph_nx.community_dict_to_in_list(d, list_dicts)
    # print(d)
    # print('---------------')
    # print(r0)
    # print('---------------')
    # print(r1)
    #===========================================================================
    print(edge_file)
    g=CliqueGraph_nx(edge_file, node_tags)
    print(type(g))
    print(g.modularity_maximization())