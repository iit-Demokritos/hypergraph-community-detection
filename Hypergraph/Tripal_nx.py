import re
import community
import scipy as sp
import networkx as nx
from Utils.FormatHandlers import parse_tripal_as_nodes
class Tripal_nx(nx.Graph):
    """
    Class tripal_nx extending networkx Graph class, processes tripal results
    """
    
    def __init__(self, tripalFile, weighted=False, node_tags=None):
        """
        Constructor, create graph from tripal results
        
        Parameters
        ----------
        tripalFile: string
            file as created from tripal algorithm,
            format: hyperedge_i hyperedge_j similarity
        weighted: boolean, optional, !NOT SUPPORTED!
            if True, use similarities as weights, unweighted graph
        
        """
        nx.Graph.__init__(self)
        
        self.weighted=False
        
        Tripal_nx.createTripalGraph(self, tripalFile, weights=weighted)
        

        self.node_tags=['U', 'T', 'L'] #default for tripal results
    
    @staticmethod
    def createTripalGraph(g, tripalFile, weights=False):
        """create graph from tripal results
        
        Parameters
        ----------
        g: networkx graph or None
            used as base graph to append to
        tripalFile: string
            file as created from tripal algorithm,
            format: hyperedge_i hyperedge_j similarity
        weights: boolean, optional, !NOT SUPPORTED!
            if True, use similarities as weights, unweighted graph
        
        Return
        ------
        g:networkx undirected graph
        """
        g.weighted=weights
        
         
        if g.weighted:
            print('...weighted edges are not in use anywhere...do not use')
            g.add_weighted_edges_from(parse_tripal_as_nodes(tripalFile, weights))
        else:
            with open(tripalFile) as f:
                for line in f:
                    els=line.split()
                    g.add_edge(els[0],els[1])
            
            del(tripalFile)
        
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
        comm_list: list
            community list where indexes correspond to 
            0:users_num , in sorted order by user id
            users_num+1: tags_num, in sorted order by tag id
            tags_num+1: links_num, in sorted order by link id
        matching: list of dictionaries
            *_id: community label dict for each type, Users, Tags, Links
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
        self.__class__=Tripal_nx
        
        
        matching={'U':{}, 'T': {}, 'L':{}}
        
        for h_e in d.keys():
            u,t,l=map(int, re.split('U|T|L',h_e)[0:3])
            
            if u in matching['U'].keys():
                matching['U'][u].add(d[h_e])
            else:
                matching['U'][u]=set([d[h_e]])
            
            if t in matching['T'].keys():
                matching['T'][t].add(d[h_e])
            else:
                matching['T'][t]=set([d[h_e]])
            
            if l in matching['L'].keys():
                matching['L'][l].add(d[h_e])
            else:
                matching['L'][l]=set([d[h_e]])

        for t in matching:
            for k in matching[t]:
                matching[t][k]=list(matching[t][k])
                
        comm_list=[]
        for t in matching:
            for k in matching[t]:
                matching[t][k]=list(matching[t][k])
            comm_list.extend([matching[t][n] for n in sorted(matching[t].keys())])
            
        return comm_list, matching, self.node_tags, q
    
#===============================================================================
# TEST
#===============================================================================
if __name__=='__main__':
    g=Tripal_nx('/home/ak/Documents/REVEAL_doc/hypergraph_simple_graph_embedding_clustering_java_and_datasets/synthetic_caveman/results/similaritiesCaveman.data')
    print(type(g))
    l,d,t,q=g.modularity_maximization()
    print(l)
    print(d)
    print(q)
    print(t)