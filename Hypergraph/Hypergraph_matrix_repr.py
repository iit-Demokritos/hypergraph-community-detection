'''
Created on Mar 21, 2016

@author: ak
'''
import gc
import heapq
from compiler.ast import flatten
import sklearn.cluster
import sklearn.manifold as skmanifold
import scipy as sp
import scipy.sparse as spsp
import scipy.sparse.linalg as spla
from collections import Counter
from Utils.FormatHandlers import to_continuous_mapping, update_edge_list, check_duplicate_indices
from Utils.JsonHandler import json_writer
from Utils.Timer import Timer
from Utils.Method_overrides_customizations import spectral_embedding

class Hypergraph_matrix_repr:
    """
    Class representing k-uniform hypergraphs as sparse matrices
    """
    def __init__(self, edge_list, weights=None, node_tags=None):
        """Constructor
        
        Parameters
        ----------
        edge_file: list of lists
            list of lists containing hyperedge vertices (n0, n1, n2, ..., nk) 
        weights: str, optional
            None: no weights, default 1
            'occurrences': weights as normalized occurrences of each tag-resource
            array: hyperedge weights ndarray
        node_tags: list of strings, optional
            list of node tags in order of appearance in hyperedge edge_file
            default=['P1', 'P2', 'P3', ...] 
        """
        if node_tags:
            self.node_tags=node_tags
        else:
            self.node_tags=['P'+str(i) for i in range(len(edge_list[0]))]
        
            
        self.edge_list, self.mapping_id_to_index, self.mapping_index_to_id, r=update_edge_list(edge_list)
        
        if len(sp.array(edge_list).shape)>1:
            self.k=sp.shape(sp.array(edge_list))[1]
        else:
            self.k=max(map(len,edge_list))
            
        #weights check
        if isinstance(weights, str):
            if weights=='occurrences':
                self.weighted=True
                self.weights=self.__occurrences_weights();
        elif weights is None:
            self.weighted=False
            self.weights=sp.ones(sp.array(self.edge_list).shape[0])
        else:
            self.weighted=weights.shape[0]==sp.array(self.edge_list).shape[0]
            self.weights=weights
        
        self.Theta=None
        self.rankFunc=None
        self.incidence_matrix_timer=-1.0
        self.laplacian_timer=-1.0
        self.eigs_timer=-1.0
        self.clustering_timer=-1.0
        
    def __occurrences_weights(self):
        """Computes weights of each hyperedge according to the number of occurrences of tag-resource pairs
         (i.e. positions 1 and 2 of each tripartite hyperedge[starting from 0])
         
         w(i)=#occurences_tag_resource_part/sum(weights) for i in |E|
         
         Returns
         -------
         w: ndarray
             vector of hyperedge weights in the order of hyperedge list
        """
        e_l_t=map(lambda x: tuple(x), self.edge_list[:,1:])
        c=Counter(e_l_t)
        w=sp.array([c[i] for i in e_l_t])
        
        return w/float(sum(c.values()))

    
    def get_times(self):
        """Returns runtimes of various operations, -1 if not computed
        
        Returns
        -------
        d: dict
            dictionary with time data (in seconds)
        """
        d={'incidence_matrix_computation_time': self.incidence_matrix_timer,
           'laplacian_computation_time': self.laplacian_timer,
           'eigenvector_computation_time': self.eigs_timer,
           'clustering_computation_time': self.clustering_timer}
        return d
            
    def vertex_degrees(self):
        """Computes vertex degrees diagonal matrix
        d(v)=sum(w(e)), where e in E, v in e
        
        Returns
        -------
        d_v: sparse diagonal matrix
            sparse diagonal vertex degree matrix
        """
        return spsp.diags(sp.bincount(self.edge_list.flatten(), weights=sp.array([[i]*self.k for i in self.weights]).flatten()))
    
    def edge_degrees(self):
        """Computes edge degrees diagonal matrix
        delta(e)=|e|
        
        Returns
        -------
        delta_e: sparse diagonal matrix
            sparse diagonal edge degree matrix
        """
        return spsp.diags([self.k]*sp.shape(self.edge_list)[0])
    
    def incidence_matrix(self):
        """Computes incidence matrix of size |V|*|E|
        h(v,e)=1 if v in e
        h(v,e)=0 if v not in e
        
        Returns
        -------
        H: sparse incidence matrix
            sparse incidence matrix of size |V|*|E|
        """
        with Timer() as t_in:
            H=spsp.lil_matrix((sp.shape(sp.unique(self.edge_list.flatten()))[0], sp.shape(self.edge_list)[0]))
        
            it=sp.nditer(self.edge_list, flags=['multi_index', 'refs_ok'])
            while not it.finished:
                H[it[0], it.multi_index[0]]=1.0
                it.iternext()
        
        self.incidence_matrix_timer=t_in.secs
        return H
    
    def adjacency_matrix(self):
        """Computes hypergraph adjacency matrix of size |V|*|V|
        
        Returns
        -------
        A: sparse adjacency matrix
            sparse adjacency matrix of size |V|*|V|
        """
        H=self.incidence_matrix().tocsr()
        W=self.weight_matrix().tocsr()
        Dv=self.vertex_degrees().tocsr()
        
        return H*W*(H.transpose())-Dv
    
    def edge_adjacency_matrix(self):
        """Computes hypergraoh edge adjacency matrix of size |E|*|E|
        
        Returns
        -------
        A: sparse adjacency matrix
            sparse adjacency matrix of size |E|*|E|
        """
        H=self.incidence_matrix().tocsr()
        #W=self.weight_matrix().tocsr()
        De=self.edge_degrees().tocsr()
        
        return H.T*H-De
    
    def weight_matrix(self):
        """Constructs hyperedge weight matrix
        
        Returns
        -------
        w: sparse diagonal matrix
            sparse diagonal hyperedge weight matrix
        """
        return spsp.diags(self.weights)

    def theta_matrix(self):    
        """Computes theta matrix
        
        Returns
        -------
        theta: sparse matrix
            sparse theta matrix
        """
        Dv=self.vertex_degrees().tocsc()
        H=self.incidence_matrix().tocsc()
        W=self.weight_matrix().tocsc()
        De=self.edge_degrees().tocsc()
        
        Dv_rec_inv=spla.inv(Dv).sqrt()
            
        self.Theta=Dv_rec_inv*H*W*(spla.inv(De))*(H.transpose())*Dv_rec_inv
        
        return self.Theta
        
    def ranking_function(self,y,m):
        """Creates ranking function
        Parameters
        ----------
        y: ndarray
            query vertex vector
        m: float
            tradeoff
        Returns 
        -------
        f: ndarray
            ranking vector
        """
#===============================================================================
# RUNS OUT OF MEMORY
#===============================================================================
#===============================================================================
#         if not self.rankFunc:
#             print('...computing Dv...')
#             Dv=self.vertex_degrees().tocsc()
# 
#             print('...computing H...')
#             H=self.incidence_matrix().tocsc()
#             
#             print('...computing W...')
#             W=self.weight_matrix().tocsc()
#             
#             print('...computing De...')
#             De=self.edge_degrees().tocsc()
#             
#             print('...computing Dv rec inv...')
#             Dv_rec_inv=spla.inv(Dv).sqrt()
#             
#             print('...computing Theta...')
#             Theta=Dv_rec_inv*H*W*(spla.inv(De))*(H.transpose())*Dv_rec_inv
#             
#             print('...deleting unnecessary matrices...')
#             del(Dv)
#             del(H)
#             del(W)
#             del(De)
#             del(Dv_rec_inv)
#             gc.collect()
#             
#             Theta=Theta.tocsc()
#             
#             print('...computing rankFunc...')
#             self.rankFunc=spla.inv((spsp.eye(*sp.shape(Theta)).tocsc()-(1/(1+m))*Theta).tocsc())
#             
#             del(Theta)
#             gc.collect()
#             
#         return self.rankFunc*y
#===============================================================================
        if self.Theta is None:
            self.theta_matrix()
        if self.rankFunc is None:   
            self.rankFunc=spsp.eye(*sp.shape(self.Theta))-(1.0/(1.0+m))*self.Theta
        f=spla.spsolve(self.rankFunc, y, use_umfpack=True)
        return f
            
    def laplacian(self):
        """Computes hypergraph laplacian
        Delta=I-Theta,
        Theta=Dv^-1/2 H W De^-1 H^T Dv^-1/2
        
        Returns
        -------
        Delta: sparse matrix
            hypergraph laplacian
        """
        
        
        with Timer() as t_l:
            
            Theta=self.theta_matrix()
            Delta=spsp.eye(*sp.shape(Theta))-Theta

        self.laplacian_timer=t_l.secs
        
        return Delta
    
    def laplacian_eigs(self, k=6, type='SM',filename=None, minTol=1e-23, **kwargs):
        """Computes eigenvectors of laplacian
        
        Parameters
        ----------
        k: int, optional
            number of eigenpairs
        type: str, optional
            type of eigenpairs, as specified in scipy.sparse.linalg.eigs documentation, or 'LNZ' for lowest non zero
        filename: str, optional
            if filename exists, save min eigenvalue, min eigenvector, all used eigenvalues, all used eigenvectors in json format
        kwargs: named arguments to pass to scipy.sparse.linalg.eigs function
        Returns
        -------
        eigenvals: ndarray
            array of k eigenvalues
        eigenvecs: ndarray
            array of k eigenvectors
        """
        min_dict={}
        
        lap=self.laplacian().tocsc()
        if k>=lap.shape[0]:
            k=lap.shape[0]-2
        if type=='LNZ':
            
            with Timer() as t_eig:
                vals,vecs=spla.eigs(lap,k=k, which='SM', **kwargs)
            
            #DBG
            print(vals)
            #sort vals and vecs
            sorted_eigenvals_indices=sp.argsort(vals)
            vals=sp.array([vals[i] for i in sorted_eigenvals_indices])
            vecs=sp.array([vecs[:,i] for i in sorted_eigenvals_indices]).T
            
            #DBG
            print(vals)
            print(sorted_eigenvals_indices)
            
            self.eigs_timer=t_eig.secs
            
            vals_lnz_indices=[i for i in range(len(vals)) if vals[i]>minTol]
            
            used_vals=sp.array([vals[i] for i in vals_lnz_indices])
            used_vecs=sp.array([vecs[:,i] for i in vals_lnz_indices]).T
            
            #DBG
            print('******eigendata:')
            print(used_vals)
            print(min(used_vals))
            if filename:
                min_dict['min_eigenval_used']=sp.real(min(used_vals)).tolist()
                min_dict['min_eigenvec_used']=sp.real(used_vecs[:,sp.argmin(used_vals)]).tolist()
                print('-----------eigenvec_len:')
                print(sp.shape(sp.real(used_vecs[:,sp.argmin(used_vals)]).tolist()))
                min_dict['eigenvals_used']=sp.real(used_vals).tolist()
                min_dict['eigenvecs_used']=sp.real(used_vecs).tolist()
                json_writer(min_dict, filename)

            self.__isPSD(lap,k)
            self.__test_eigenpairs(vals, vecs, lap)
            
            return used_vals,used_vecs
        else:
            with Timer() as t_eig:
                vals,vecs=spla.eigs(lap, k=k, which=type, **kwargs)
            
            sorted_eigenvals_indices=sp.argsort(vals)
            vals=sp.array([vals[i] for i in sorted_eigenvals_indices])
            vecs=sp.array([vecs[:,i] for i in sorted_eigenvals_indices]).T
            
            self.eigs_timer=t_eig.secs
            
            if filename:
                min_dict['min_eigenval_used']=sp.real(min(vals)).tolist()
                min_dict['min_eigenvec_used']=sp.real(vecs[:,sp.argmin(vals)]).tolist()
                min_dict['eigenvals_used']=sp.real(vals).tolist()
                min_dict['eigenvecs_used']=sp.real(vecs).tolist()
                json_writer(min_dict, filename)

            return vals,vecs
      
    @staticmethod  
    def __test_eigenpairs(vals, vecs, matrix):
        """Test correct computation of eigenpairs
        
        Parameters
        ----------
        vals : vector of eigenvalues
        vecs : eigenvector matrix, vecs[:,i] corresponding to eigenvalue vals[i]
        matrix : matrix from where eigenpairs came from
        Returns
        -------
        res : list of (bool, float)
            truth value of eigenpair corectness along with error in computation
        """ 
        res=[]
        for i in range(len(vals)):
            A=matrix*vecs[:,i]
            B=vals[i]*vecs[:,i]
            res.append((any(sp.equal(A,B)), sp.sum(A-B)))
        print('-----------laplacian test:')
        print(res)
        return res
    
    @staticmethod
    def __isPSD(A,k, tol=1e-10):
        E,V = spla.eigsh(A,k)
        print('-----------PSD test:')
        print(sp.all(E > -tol))
        return sp.all(E > -tol)
    
    
    def spectral_clustering(self, clusters_n, k=6, type='SM', embed_type='custom', **kwargs):
        """Performing k-means spectral clustering on laplacian eigenvectors via scikit-learn kmeans algo
        
        Parameters
        ----------
        clusters_n: int
            number of clusters
        k: int, optional
            num of eigenvectors to base kmeans
        type: str, optional
            type of eigenvectors to use for kmeans, as specified in laplacian_eigs
        embed_type: str, optional
            choices: 'custom' - perform embedding using hypergraph laplacian and custom implemented embedding
                     'sklearn_laplacian' - perform embedding using modified sklearn.spectral.embedding using the hypergraph laplacian
                     'sklearn_adjacency' - perform embedding using original sklearn.spectral.embedding using the hypergraph adjacency matrix
            default is 'custom'
        kwargs: named arguments to pass to laplacian eigs function

        Returns
        -------
        centroid: ndarray of shape (k, n_features)
        label: ndarray of shape (n_samples,)
        label_dict: dictionary
            dictionary containing {partiteName: { id: communityId , ...} , ... }
        node_tags: list of str
            order of partites, as found in hyperedges
        inertia: float
        """
        if embed_type=='sklearn_laplacian':
            f=None
            if 'filename' in kwargs:
                f=kwargs.pop('filename')
            if 'minTol' in kwargs:
                kwargs.pop('minTol')
            if 'maxiter' in kwargs:
                kwargs.pop('maxiter')
            eigenvecs=spectral_embedding(self.laplacian(),clusters_n, **kwargs)
            
            #===================================================================
            # #DBG
            # print(eigenvecs)
            # print(eigenvecs.min())
            # print(eigenvecs.max())
            # print(eigenvecs.mean())
            #===================================================================
            if f:
                json_writer({'eigenvecs_used':sp.real(eigenvecs).tolist()}, f)
             
        elif embed_type=='sklearn_adjacency':
            f=None
            if 'filename' in kwargs:
                f=kwargs.pop('filename')
            if 'minTol' in kwargs:
                kwargs.pop('minTol')
            if 'maxiter' in kwargs:
                kwargs.pop('maxiter')
            eigenvecs=skmanifold.spectral_embedding(self.adjacency_matrix(), clusters_n, **kwargs)
            
            #===================================================================
            # #DBG
            # print(eigenvecs)
            # print(eigenvecs.min())
            # print(eigenvecs.max())
            # print(eigenvecs.mean())
            #===================================================================
            if f:
                json_writer({'eigenvecs_used':sp.real(eigenvecs).tolist()}, f)
                
        else:
            eigenvecs=self.laplacian_eigs(k,type, **kwargs)[1]
        
        with Timer() as t_cl:
            cen, lab, inert=sklearn.cluster.k_means(eigenvecs, clusters_n)
        
        self.clustering_timer=t_cl.secs
        
        label_dict=self._community_vector_match(lab)

        return cen, lab, label_dict, self.node_tags, inert
    
    def _community_vector_match(self, community_array):
        """Matches vector initial ids to labels
        
        Parameters
        ----------
        community_array : list
            array of community labels, len()==num_of_vertices
        
        Returns
        -------
        label_matching_dict: list of dictionaries
            {id: label, } dictionary
        """

        label_matching_dict={}
        for ind,m in enumerate(self.mapping_index_to_id):
            t_m={}
            for i in range(len(community_array)):
                if i in m:
                    t_m[m[i]]=community_array[i]
            label_matching_dict[self.node_tags[ind]]=t_m
        
        return label_matching_dict
    
    
    
    def hypergraph_vertex_ranking(self,y, tradeoff=0.5, top='all', labels=False, return_query_v=True, concentrate_on='all'):
        """Function for vertex ranking on hypergraph based on selected modality with which weights are created
            see: Xu, J.; Singh, V.; Guan, Z. & Manjunath, B. S. (2012), Unified hypergraph for image ranking in a multimodal context., in 'ICASSP' , IEEE, , pp. 2333-2336 . 
            
        Parameters
        ----------
        y: array
            query vertex array, 1 at query vertex, len=|V|
        tradeoff: float, optional
            tradeoff value for scoring function
        top: str or int, optional
            if 'all', return all results,
            if int, return top x results
        labels: bool, optional
            if True, return tuples of (ranking, n_id, type)
            if False, return array of len |V| with rankings
        return_query_v: bool, optional
            if True(default) returns query vertex ranking as well (highest)
            if False discard query vertex ranking at return
        contentrate_on: str or int or list or str/int, optional
            all - all node types (default)
            type - as in types
            int - index of type in hyperedge
            list of str/int - list of above
        Returns
        -------
        f: array
            array of vertex rankings, len=|V| including self
            if labels, array consists of tuples (ranking, id, type)
        
        """
        
        #rankings array
        f=self.ranking_function(y, tradeoff)
        
        if concentrate_on=='all':
            t_c_l=range(len(self.node_tags))
        elif isinstance(concentrate_on,list):
            t_c_l=map(lambda x: x if isinstance(x,int) else self.node_tags.index(x), concentrate_on)
        else:
            t_c_l=[concentrate_on if isinstance(concentrate_on,int) else self.node_tags.index(concentrate_on)]
        
        t_c_l_n=list(set(range(len(self.node_tags)))-set(t_c_l))
        
        #top and label extraction
        if labels==False:
            
            if not return_query_v:
                q_d=list(sp.where(sp.array(y)>0)[0])
            f=sp.delete(f,q_d+flatten([self.mapping_index_to_id[i].keys() for i in t_c_l_n]))
            
            if isinstance(top, str) and top=='all':
                return f
            else:
                return heapq.nlargest(top, f)
        
        else:
            if isinstance(top,str):
                top=len(f)
            
            f_q=f
            if not return_query_v:
                q_d=list(sp.where(sp.array(y)>0)[0])
            f_q=sp.delete(f_q,q_d+flatten([self.mapping_index_to_id[i].keys() for i in t_c_l_n]))

            t_f=sorted(list(heapq.nlargest(top,set(f_q))))[::-1]
            
            f_q_s=sorted(list(heapq.nlargest(top, f_q)))[::-1]            
            
            inds=sp.array(flatten([sp.where(f==i)[0].tolist() for i in t_f]))
            
            labels=[(self.mapping_index_to_id[i][ind], self.node_tags[i]) for ind in inds for i in t_c_l if ind in self.mapping_index_to_id[i]]    
            
            return [(i[0],i[1][0],i[1][1]) for i in zip(f_q_s,labels)]
        

#===============================================================================
# TEST
#===============================================================================
if __name__=='__main__':
    #===========================================================================
    # edge_list=sp.array([
    #            [1,1,3],
    #            [1,4,5],
    #            [1,1,7],
    #            [2,8,9],
    #            [2,1,3],
    #            [7,8,9],
    #            [15,8,9]])
    #===========================================================================
    edge_list=sp.array([
               [1,1,3],
               [1,1,4],
               [1,1,7],
               [2,8,3],
               [2,1,3],
               [2,5,9],
               [2,12,9],
               [100,200,300]])
    #===========================================================================
    # list_dicts=[{0:'user0', 1:'user1'},
    #             {11:'tag1', 12:'tag2'},
    #             {23:'res3', 24:'res4', 25:'res5', 28:'res8'}]
    #===========================================================================
    tag_list=['user', 'tag', 'res','what']
      
    g=Hypergraph_matrix_repr(edge_list)
    print(edge_list)
    print(g.mapping_id_to_index)
    print(g.mapping_index_to_id)
    print(g.edge_list)
    print(type(g.edge_list))
    print('---------------')
    print(g.weight_matrix().toarray())
    print(g.adjacency_matrix().toarray())
    print('---------------')
    print(g.vertex_degrees())
    print('---------------')
    print(g.edge_degrees())
    print('---------------')
    print(g.incidence_matrix())
    print('---------------')
    print(g.laplacian().toarray())
    print('---------------')
    print(g.laplacian_eigs(type='SM',maxiter=10**10,tol=1e-23))
    print('---------------')
    print(g.laplacian_eigs(maxiter=10**10,tol=1e-23))
    print('---------------')
    print(g.laplacian_eigs()[1].shape)
    print('---------------')
    s=g.spectral_clustering(3,k=3, embed_type='custom', filename='/home/ak/workspace/test.json')
    s=g.spectral_clustering(3,k=3, embed_type='sklearn_adjacency', filename='/home/ak/workspace/test.json')
  
    s=g.spectral_clustering(3,k=3, embed_type='sklearn_laplacian', filename='/home/ak/workspace/test.json')
  
    print('here:')
    print(s)
    print(g._community_vector_match(s[1]))
  
    print(g.get_times())
    print(['n1','n2','n1','n4','n6','n8','n3','n5','n7','n9'])
    print(g.adjacency_matrix().toarray())
    print(edge_list)
    print('\n*****************************************************************************************\n')
    g=Hypergraph_matrix_repr(edge_list, weights='occurrences')
    print(g.weight_matrix().toarray())
    print(g.theta_matrix().shape)
    q_v=sp.zeros(13)
    q_v[0]=1
    print(g.theta_matrix().toarray())
    print("---------------------------")
    print(q_v.reshape(-1,1))
    print(g.edge_list)
    print(g.mapping_index_to_id)
    print(g.hypergraph_vertex_ranking(q_v, 0.5, top='all', labels=True, return_query_v=False))
    print(g.weight_matrix().diagonal())
    print(edge_list)
    
