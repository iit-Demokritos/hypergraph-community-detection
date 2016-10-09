'''
Created on Aug 19, 2016

@author: ak
'''
import sys
import os
sys.path.append(os.getcwd())

import re
import json
import pandas as pd
import scipy as sp
import sklearn.metrics as sklm
import math
import sortedcontainers as sc
from scipy.special import binom
from collections import Counter
from compiler.ast import flatten

from JsonHandler import *
from Utils.FormatHandlers import *

#===============================================================================
# NMI
#===============================================================================

def NMI_sklearn(algo_matchings, ground_truth_matchings, node_tags=None):
    """
    Computes NMI score (sklearn function wrapper)
    
    Parameters
    ----------
    algo matchings: list or dict
        community assignment list or dictionary with elements for each partite, containing {nodeId: communityId,...} assignments
        as returned from the tested algorithm
    ground_truth_matchings: list or dict
        ground truth assignment list or dictionary with elements for each partite, containing {nodeId: communityId,...} assignments
        regarded as ground truth assignments
    node_tags: list of str, optional
        list of partite names
        
    Returns
    -------
    NMI: float
        NMI score
    """
    if isinstance(algo_matchings, list):
        algo_ar=algo_matchings
    else:
        algo_ar=community_dict_to_community_array(algo_matchings, node_tags=node_tags)
    
    if isinstance(ground_truth_matchings, list):
        gt_ar=ground_truth_matchings
    else:
        gt_ar=community_dict_to_community_array(ground_truth_matchings, node_tags=node_tags)
    
    NMI=sklm.normalized_mutual_info_score(gt_ar, algo_ar)

    return NMI


def NMI(hfile, algo_comm_array, ground_truth_comm_array, node_tags=None):
    """
    compute NMI
    in case of overlapping communities each node is re-counted for each distinct community Id assigned to it
    
    Parameters
    ----------
    hfile: list of hyperedges
    algo_comm_array: list/array/dictionary of algorithm community assignments
    ground_truth comm_array:  list/array/dictionary of ground truth community assignments
    node_tags: list of str, optional
        list of partite names
        
    Returns
    -------
    NMI: NMI score
    """
    df=to_dataframe(hfile, algo_comm_array, ground_truth_comm_array, node_tags)
    L=df.loc[:,'ground_truth_comm_id'].unique()    #labels
    H=df.loc[:,'assigned_comm_id'].unique()    #found comms
    n=sum([len(df[df['partite_id']==i]['nId']) for i in df.loc[:,'partite_id'].unique()]) #total number of nodes
    dfLcounts=df.groupby('ground_truth_comm_id')['nId'].count() #number of nodes per label
    dfHcounts=df.groupby('assigned_comm_id')['nId'].count() #number of nodes per found community
    dfHLcounts=df.groupby(['assigned_comm_id','ground_truth_comm_id'])['nId'].count()
    print(df[df['partite_id']==1])
    print('----------------------')
    print(dfLcounts)
    print(dfHcounts)
    print(dfHLcounts)
    
    Hh=0
    for h in H:
        Hh+=float(dfHcounts.loc[h])*math.log(  float(dfHcounts.loc[h])/float(n) )
        
    Hl=0
    for l in L:
        Hl+=float(dfLcounts.loc[l])*math.log(  float(dfLcounts.loc[l])/float(n) )
    
    MI=0
    for h in H:
        for l in L:
            try:
                #===============================================================
                # print(n)
                # print(dfHLcounts.loc[h][l])
                # print(float(n*dfHLcounts.loc[h][l]))
                # print(dfHcounts.loc[h])
                # print(dfLcounts.loc[l])
                # print(dfHcounts.loc[h]*dfLcounts.loc[l])
                #===============================================================
                MI+=float(dfHLcounts.loc[h][l])*math.log(float(n*dfHLcounts.loc[h][l])/float(dfHcounts.loc[h]*dfLcounts.loc[l]))
                #===============================================================
                # print(MI)
                # print('---')
                #===============================================================
            except KeyError:
                MI+=float(0)


    NMI=MI/(sp.sqrt(Hh*Hl))
    
    print('NMI %f'%NMI)
    print('MI %f'%MI)
    print('Hh %f'%Hh)
    print('Hl %f'%Hl)
    
    return NMI

#===============================================================================
# PRECISION/RECALL BASED MEASURES
#===============================================================================
def recall(nid,type_n, comm_dict, rankings,concentrate_on='all',types=['user', 'tag', 'resource'], edges=None):
    """Computes recall of ranked nodes belonging in the same community as the query node
    
    Parameters
    ----------
    nid: int
        query node id
    type_n: str
        type of nid
    comm_dict: dict or community list
        id: community_id matchings as returned from community detection algorithms
            or community list as returned from community detection algorithms.
            !In case of community list must provide edge list as well at optional argument edge_list 
    rankings: list
        rankings as return from ranking algorithm (with labels)
    contentrate_on: str or list of str, optional
        all - all node types (default)
        type - as in types
        int - index of type in hyperedge
        list of str/int - list of above
    types: list
        list of node types as strings
    edges: hyperedge list, optional
        in use when community array is passed at comm_dict
    Returns
    -------
    percentage: float
        percentage of nodes in the same community
    n: list
        list of (rankings, nid, type) of nodes in same community
    """
    if isinstance(comm_dict, list) and not isinstance(comm_dict[0],dict) and edges:
        comm_dict,_=community_array_to_community_dict(comm_dict, edges)
    
    if concentrate_on=='all':
        t_c_l=types
    elif isinstance(concentrate_on,list):
        t_c_l=concentrate_on
    else:
        t_c_l=concentrate_on
    
    c_id=comm_dict[type_n][nid]
    
    in_comm=[]
    
    c_id=c_id if isinstance(c_id, list) else [c_id]
    
    for t_c in t_c_l:
        for i_c_id in c_id:
            in_comm.extend(tuple(i) for i in rankings if ((i[1] in comm_dict[t_c]) and (comm_dict[t_c][i[1]]==i_c_id if isinstance(comm_dict[t_c][i[1]],int) else (i_c_id in comm_dict[t_c][i[1]]))))


    in_comm=list(set(in_comm))
    return len(in_comm)/float(len(rankings)), in_comm

def precision(nid, type_n, comm_dict, rankings,concentrate_on='all',types=['user', 'tag', 'resource'], edges=None):
    """Computes precision of ranked nodes belonging in the same community as the query node
    
    Parameters
    ----------
    nid: int
        query node id
    type_n: str or int
        type of index of hyperedge element
    comm_dict: dict or community list
        id: community_id matchings as returned from community detection algorithms
            or community list as returned from community detection algorithms.
            !In case of community list must provide edge list as well at optional argument edge_list 
    rankings: list
        rankings as return from ranking algorithm (with labels)
    contentrate_on: str or int or list or str/int, optional
        all - all node types (default)
        type - as in types
        int - index of type in hyperedge
        list of str/int - list of above
    types: list
        list of node types as strings
    edges: hyperedge list, optional
        in use when community array is passed at comm_dict
    Returns
    -------
    percentage: float
        percentage of nodes in the same community
    n: list
        list of (rankings, nid, type) of nodes in same community
    """
    
    if isinstance(comm_dict, list) and not isinstance(comm_dict[0],dict) and edges:
        comm_dict,_=community_array_to_community_dict(comm_dict, edges)

    counts=[Counter(flatten(comm_dict[d].values())) for d in comm_dict]
    
    if concentrate_on=='all':
        t_c_l=types
    elif isinstance(concentrate_on,list):
        t_c_l=concentrate_on
    else:
        t_c_l=concentrate_on
    
    #community of specified node
    c_id=comm_dict[type_n][nid]
    
    c_id=c_id if isinstance(c_id, list) else [c_id]
    
    in_comm=[]
    
    for t_c in t_c_l:
        for i_c_id in c_id:
            in_comm.extend(tuple(i) for i in rankings if ((i[1] in comm_dict[t_c]) and (comm_dict[t_c][i[1]]==i_c_id if isinstance(comm_dict[t_c][i[1]],int) else (i_c_id in comm_dict[t_c][i[1]]))))

    in_comm=list(set(in_comm))
    return len(in_comm)/float(sum([counts[j][i] for i in c_id for j in t_c_l])), in_comm

def F1(p,r):
    """
    Computes F1 measure (precision-recall harmonic mean)
    
    Parameters
    ----------
    p: float
        precision
    r: float
        recall
    
    Return
    ------
    F1: float
        F1 measure
    """
    return 2.0*((p*r)/float(p+r)) if p+r!=0 else 0.0

#===============================================================================
# CONDUCTANCE
#===============================================================================
def conductance(comm, hyperedges, node_tags=None):
    """Compute clustering conductance measure
    
    Parameters
    ----------
    comm: list or dict
        community list of community dictionary
    hyperedges: list of lists
        list of hyperedges
    node_tags: list of str, optional, if comm is a dictionary, specify correct order of keys
        list of partite labels, as found in the hyperedges
    
    Returns
    -------
    scores: dictionary
        conductance scores 
    """
    if isinstance(comm, list):
        comm_dict,node_tags=community_array_to_community_dict(comm, hyperedges)
    else:
        if not node_tags:
            node_tags=partite_order(hyperedges, comm)
        comm_dict=comm
    
    out_edges={}
    degs={}
    _he_comm_list=[]
    for he in hyperedges:
        _he_comm_list=[comm_dict[node_tags[ind]][n] for ind,n in enumerate(he)] #nominator
        if any(isinstance(x, list) for x in _he_comm_list): #overlapping case
            _he_comm_counts=dict(Counter(flatten(_he_comm_list)))
            for n in _he_comm_counts:
                if _he_comm_counts[n]<len(comm_dict):
                    if n not in out_edges:
                        out_edges[n]=1
                    else:
                        out_edges[n]=1+out_edges[n]
        elif len(sp.unique(_he_comm_list))>1: #non overlapping case
            for n in sp.unique(_he_comm_list):
                if n not in out_edges:
                    out_edges[n]=1
                else:
                    out_edges[n]=1+out_edges[n]
        for n in flatten(_he_comm_list): #denominator
            if n not in degs:
                degs[n]=1
            else:
                degs[n]=1+degs[n]
    
    scores={}
    for c in degs:
        other_deg=sp.array([v for k,v in degs.items() if k!=c]).sum()
        scores[c]=float(out_edges[c])/min([float(degs[c]), float(other_deg)])
        
    return scores

def network_community_profile(comm, hyperedges, node_tags=None ,verbose=False):
    """Computes network community profile (NCP), as found in:
    Jure Leskovec, Kevin J. Lang, and Michael Mahoney. 
    2010. 
    Empirical comparison of algorithms for network community detection. 
    In Proceedings of the 19th international conference on World wide web (WWW '10). 
    ACM, New York, NY, USA, 631-640. 
    DOI=http://dx.doi.org/10.1145/1772690.1772755
    
    Parameters
    ----------
    comm: list or dict
        community assignment as list or dictionary
    hyperedges: list of lists
        community list of community dictionary
        list of hyperedges
    node_tags: list of str, optional, if comm is a dictionary, specify correct order of keys
        list of partite labels, as found in the hyperedges
    verbose: bool, optional
        if true, return verbose network community profiles ncp_v
        default=False
        
    Returns
    -------
    ncp: sorted dictionary
        network community profile in dictionary form
    nvp_v: sorted dictionary, only if verbose=True
        dictionary containing all measured conductance scores per community size
    """
    
    ncp=sc.SortedDict()
    if verbose:
        ncp_v=sc.SortedDict()
        
    comm=comm if isinstance(comm, list) else [comm]
    
    for comm_res in comm:
        scores=conductance(comm_res, hyperedges, node_tags)
        sizes=community_size(comm_res)
        
        for c in scores:
            csize=sizes[c]
            if csize in ncp:
                ncp[csize]=min(ncp[csize],scores[c])
            else:
                ncp[csize]=scores[c]
                
            if verbose:
                if csize in ncp_v:
                    ncp_v[csize].append(scores[c])
                else:
                    ncp_v[csize]=[scores[c]]
    
    if not verbose:
        return ncp
    else:
        return ncp, ncp_v
    
    

def community_size(comm):
    """Computes community sizes for each community
    
    Parameters
    ----------
    comm: list or dict
        community array or dictionary
    
    Returns
    -------
    sizes: dict
        dictionary of community sizes
    """
    comm_ar=[]
    if isinstance(comm, dict):
        for d in comm:
            comm_ar.extend(comm[d].values())
    else:
        comm_ar=comm
        
    return dict(Counter(flatten(comm_ar)))

def community_size_distribution(comm):
    """Computes community sizes for each community
    
    Parameters
    ----------
    comm: list or dict
        community array or dictionary
    
    Returns
    -------
    distr: tuple
        (mean, std)
    """
    c=community_size(comm).values()
    
    return (sp.mean(c), sp.std(c))
    
    
#===============================================================================
# TEST
#===============================================================================
if __name__=='__main__':
    hyperedges=[[1,2,3],
                [1,4,5],
                [3,4,5],
                [23,4,1],
                [10,11,3]]
    
    comm={'P1':{1:1 , 3:1, 23:2, 10:3}, 'P2':{2:1, 11:2, 4:2}, 'P3':{3:3, 5:2, 1:1}}
    
    print(conductance(comm, hyperedges))
    
    ind_list1 = [1, 1, 2, 2, 3]
    ind_list2 = [1, 1, 2, 4, 4]
    
    print(NMI_sklearn(ind_list1, ind_list2))
    
    print(network_community_profile(comm, hyperedges, verbose=True))