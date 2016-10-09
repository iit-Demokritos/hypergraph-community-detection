'''
Created on Jun 15, 2016

@author: ak
'''

import sys
import os
from warnings import warn
from operator import itemgetter
sys.path.append(os.getcwd())

import re
import json
import pandas as pd
import scipy as sp
import sklearn.metrics as sklm
import math
from scipy.special import binom

from JsonHandler import *

#===============================================================================
# -------------------------HELPER FUNCTIONS-------------------------------------
#===============================================================================
#===============================================================================
# HYPEREDGE FILE HANDLING (JSON, RAW)
#===============================================================================
def json_to_raw(filename,savepath):
    """
    Convert json hyperedges to raw format:
    n1 n2 n3
    n4 n5 n6
    ...
    
    Parameters
    ----------
    filename: json file
        json file containing hyperedges as a list of lists {key: [[n1, n2, n3], [n4, n5, n6], ... ]}
    savepath: raw filename
        name of file to save results
    """
    d=json_reader(filename).values()[0]
    print(filename.split('/')[-1].split('.')[0])
    with open(savepath+filename.split('/')[-1].split('.')[0]+'.raw','w') as f:
        for e in d:
            f.write(' '.join([str(i) for i in e])+'\n')

def raw_to_json(filename, savepath):
    """
    Convert raw data file containing hyperedges to json file
    i.e.,
    
    n1 n2 n3        {'hyperedges': [[n1, n2, n3],
    n4 n5 n6  -->                   [n4, n5, n6],
    ...                             ...]}
    
    !WARNING: node indices are integers!
    
    Parameters
    ----------
    filename: raw filename
        file containing hyperedges in raw format
    savepath: json filename
        json file to save results
    
    """
    l=parse_raw_hyperedges(filename)
    json_writer({'hyperedges':l}, savepath)


#===============================================================================
# HYPEREDGE PARSERS/READERS
#===============================================================================
def parse_hyperedges_from_file(hfile):
    """Parses hyperedges from given file
    
    Parameters
    ----------
    hfile: str
        filename
    
    Returns
    -------
    hyperedges: list of lists
        list of hyperedges
    """
    
    hyperedges=[]
    if isinstance(hfile, str):
        he,ext=os.path.splitext(hfile)
        if ext=='.json':
            hyperedges=json_reader(hfile).values()[0]
        elif ext=='.data':
            hyperedges=parse_tripal_hyperedges(hfile)
        else:
            hyperedges=parse_raw_hyperedges(hfile)
    return hyperedges


    
def parse_raw_hyperedges(hfile):
    """
    Parses raw hyperedges file into a list of lists
    
    !WARNING: node indices are integers!
    
    Parameters
    ----------
    hfile: file containing one hyperedge per line
    
    Returns
    -------
    hyperedges: list of lists containing hyperedges
    """
    
    hyperedges=[]
    with open(hfile) as f:
        for line in f:
            he=line.split()
            hyperedges.append(map(int,he))
    return hyperedges

def parse_tripal_hyperedges(tripalFile):
    """
    Parse hyperedges as output from Murata's tripal method 
        U<id1>T<id2>L<id3> similarity_score
        
    !WARNING: node indices are integers!
    
    Prameters
    ---------
    tripalFile: filename
        file output of tripal method
    
    Returns
    -------
    hyperedges: list of lists
        hyperedges as list of lists [[n1, n2, n3], [n4, n5, n6], ...]
    """
    hyperedges=[]
    with open(tripalFile) as f:
        for line in f:
            els=line.split()
            for i in [0,1]:
                hyperedges.append(map(int,re.split('U|T|L',els[i])[0:3]))

    return hyperedges

def parse_tripal_as_nodes(tripalFile, weights):
        """parses tripal results
        
        Parameters
        ----------
        tripalFile: string
            tripal results file
        weights: bool
            include similarities or not
        Returns
        -------
        tripalData: list of tuples
            list of tuples 
                [(hyperedge_i, hyperedge_j,[similarity]), ...]
        """
        tripalData=[]
        with open(tripalFile) as f:
            for line in f:
                els=line.split()
                tripalData.append((els[0], els[1], float(els[2])) if weights else (els[0], els[1]))
        
        return tripalData
    
def parse_json_hyperedges(hfile):
    """
    Parse hyperedge list from json file
        
    Parameters
    ----------
    hfile: json file
        file containing list of lists (hyperedges) as value (key is irrelevant, as long as the value of the first element is the list of hyperedges)
    
    Returns
    -------
    hyperedges: list of lists
        list of lists containing hyperedges as sets of integer nodeIds
    """
    hyperedges=json_reader(hfile).values()[0]
    
    return hyperedges

#===============================================================================
# OTHER PARSERS
#===============================================================================
def parse_community_array_from_file(algo_comm_array):
    """Parses community array from file
    
    Parameters
    ----------
    algo_comm_array: str
        filename
        
    Returns
    -------
    algo_ar: list
        list of community assignments, in sorted order by partite by node index
    """

    if isinstance(algo_comm_array, str):
        algo_ar=[]
        algocomms,ext=os.path.splitext(algo_comm_array)
        if ext=='.json':
            algo_ar=json_reader(algo_comm_array).values()[0]
        else:
            algo_ar=parse_community_array(algo_comm_array)
       
    return algo_ar


def parse_community_array(f):
    """parses raw file resulting from community assignment( community array)
    
    Parameters
    ----------
    f: str
        filename
    
    Returns
    -------
    comm_list: ndarray
        community array
    """
    comm_list=[]
    with open(f) as h_f:
        for line in h_f:
            if line[0]=='[':
                l=[]
                for i in range(len(line)):
                    if line[i]=='[':
                        line=[]
                    elif line[i]==']':
                        comm_list.append(l)
                    elif line[i]==',':
                        None
                    else:
                        l.append(int(line[i]))
            else:
                comm_list.extend(map(int, line.split(',')))
    
    return comm_list
    

#===============================================================================
# INDEXING TRANSFORMATIONS
#===============================================================================
def to_continuous_mapping(hyperedges, included=None):
    """
    Convert ids to continuous ids
    
    Parameters
    ----------
    hyperedges: list of lists
        list of lists containing hyperedges
    included: object, optional
        assign mapping of elements included in object, else do not take into account
    
    Returns
    -------
    mappings: list of dictionaries
        list of dictionaries in the form {index: id, } or {id: index, } for each partite
        ids are assigned based on the sorted order of keys_list
    mappings_range: tuple
        (start_index, end_index)
    """
    
    #extract unique ids from hyperedge list
    if len(sp.array(hyperedges).shape)==2: #uniform hypergraph
        keys_list=[sp.unique(sp.array(hyperedges)[:,i]) for i in range(sp.array(hyperedges).shape[1])]
    else:   #non uniform hypergraph
        keys_list=[]
        for he in hyperedges:
            for ind,n in enumerate(he):
                if ind>len(keys_list)-1:
                    keys_list.append(set())
                keys_list[ind].add(n)
        keys_list=map(list, keys_list)
          
    mappings_index_to_id=[]
    mappings_id_to_index=[]
    last_index=0
    for i in range(len(keys_list)):
        l=keys_list[i]
        if included!=None:
            l=[el for el in l if el in included[:,i]]
        mappings_index_to_id.append(dict(zip(range(last_index, last_index+len(l)), sorted(l))))
        mappings_id_to_index.append(dict(zip(sorted(l), range(last_index, last_index+len(l)))))
        last_index+=len(l)
    
    return(mappings_id_to_index, mappings_index_to_id, (0,last_index-1))


def update_edge_list(edge_list):
    """Create hyperedge list based on mapping
    
    Parameters
    ----------
    edge_list: list of lists
        list of hyperedges
    
    Returns
    -------
    upd_edge_list: list of tuples
        list of updated hyperedges
    mappings_id_to_index: list of dictionaries
        list of dictionaries in the form {id: index, } for partite
        ids are assigned based on the sorted order of partite keys
    mappings_index_to_id: list of dictionaries
        list of dictionaries in the form {index: id, } for partite
        ids are assigned based on the sorted order of partite keys
    r: tuple
        (start_index, end_index)
    """
    mappings_id_to_index, mappings_index_to_id,r=to_continuous_mapping(edge_list)
    upd_edge_list=sp.array([[mappings_id_to_index[i][h_e[i]] for i in range(len(h_e))] for h_e in edge_list if all([h_e[j] in mappings_id_to_index[j].keys() for j in range(len(h_e))])])
    return upd_edge_list, mappings_id_to_index, mappings_index_to_id, r

def check_duplicate_indices(edge_list):
    """Checks the hyperedge list for duplicate indices between different dimensions/partites.
    
    !WARNING: only for uniform hypergraphs!
    
    Parameters
    ----------
    edge_list: list of lists
        list of hyperedges
    
    Returns
    -------
    duplicate: boolean
        True 
            if duplicate indices between dimensions/partites
        else
        False
    """
    edge_list=sp.array(edge_list)
    columnwise_unique=sp.concatenate([sp.unique(edge_list[:,i]) for i in range(edge_list.shape[1])]).ravel()
    u, counts=sp.unique(columnwise_unique, return_counts=True)
    if any(counts>1):
        return True
    else:
        return False

def community_array_to_community_dict(comm_array, hyperedges, node_tags=None):
    """Converts community array to community dictionary
    
    Parameters
    ----------
    comm_array: list
        list of community memberships, order based on sorted indices of 'sorted' partite structure (i.e., partite order must match hyperedges partite order)
    hyperedges: list of lists
        list of hyperedges
    node_tags: list of str, optional
        partite names, default=['P1', 'P2', ...]
        
    Returns
    -------
    comm_dict: dictionary
        dictionary containing {partiteName: { id: communityId , ...} , ... }
    node_tags: list of str
        partite ids
        
    """
    if not node_tags:
        node_tags=['P'+str(i) for i in range(len(hyperedges[0]))]
            
    _,mapping_index_to_id,__=to_continuous_mapping(hyperedges)
    
    comm_dict={}
    for pind,partite in enumerate(mapping_index_to_id):
        comm_dict[node_tags[pind]]={partite[ind]:comm_array[ind] for ind in partite}
    
    return comm_dict,node_tags

def community_dict_to_community_array(comm_dict, hyperedges=None, node_tags=None):
    """Converts community dictionary to community array
    
    Parameters
    ----------
    comm_dict: dictionary
        community assignment dictionary {partiteId: {nodeId: communityId, ...} }
    node_tags: list of str, optional
        list of partiteIds in the order of appearance in hyperedges file
        if node_tags are not provided, will try to extract correct partite structure from hyperedges
    
    Return
    ------
    comm_ar: list
        list of community assignments
    """
    
    if not hyperedges and not node_tags:
        raise Exception('must provide either hyperedges or node tags')
    
    if not node_tags:    
        node_tags=partite_order(hyperedges, comm_dict)
        if not node_tags:
            warn('provide node_tags for correct execution')
        
    comm_ar=[]
    for t in node_tags:
        comm_ar.extend([comm_dict[t][n] for n in sorted(comm_dict[t].keys())])
        
    return comm_ar

def partite_order(hyperedges, comm_dict):
    """Attempt to match partite names and order of comm_dict with given hyperedge file
    
    Parameters
    ----------
    hyperedges: list of lists
        list of hyperedges of uniform hypergraph
    comm_dict: dict
        dictionary of {'partiteId':{nodeId:communityLabel, ...} , ...} mappings
    
    Returns
    -------
    node_tags: list of str
        list of correct order of partite ids or None
    """
    node_tags=[]
    hyperedges=sp.array(hyperedges)
    
    if len(hyperedges.shape)<2:
        return None
    
    for dim in range(hyperedges.shape[1]):
        for k in comm_dict:
            if set(comm_dict[k].keys())==set(hyperedges[:,dim]):
                node_tags.append(k)
    
    if len(node_tags)!=hyperedges.shape[1]:
        warn('No partite order could be derived, node ids are identical between partites, consider providing correct partite ordering')
    
    return node_tags

#===============================================================================
# OTHER FORMAT TRANSFORMATIONS
#===============================================================================
def to_dataframe(hyperedges, algo_comm, ground_truth_comm, node_tags=None):
    '''
    convert data to pandas dataframe for easy manipulation
    
    Parameters
    ----------
    hfile: list of hyperedges
    algo_comm_array: list/array/dictionary of algorithm community assignments
    ground_truth comm_array:  list/array/dictionary of ground truth community assignments
    node_tags: list of str
        list of partite names, as found in hfile. Alternatively order will be extracted from given files, but exact duplicate indices will cause conflicts, 
        default is ['P1', 'P2', ...]
    Returns
    -------
    df:pandas dataframe with columns:
        partite, nodeId, label (ground truth), community_id (assigned community)
    '''
        
    #partite count
    partitecount=len(hyperedges[0])
    
   
    if isinstance(algo_comm, dict):
        algo=algo_comm
        if not node_tags:
            node_tags=partite_order(hyperedges, algo_comm)
    else:    
        #create dictionary from community array
        algo,node_tags=community_array_to_community_dict(algo_comm,hyperedges, node_tags)


    if isinstance(ground_truth_comm, dict):
        gt=ground_truth_comm
        if set(node_tags)!=set(gt.keys()):
            warn('partite ids between ground truth matchings and algorithm matchings do not match')    
    else:
        #create dictionary from community array
        gt,_=community_array_to_community_dict(gt,hyperedges, node_tags)   

    #nodes per partite if ground truth matching
    nodesPerPartite=len(gt)/partitecount    
    
    #form dataframe
    df=[]
    for i in node_tags:
        for n in algo[i].keys():

            L=gt[i][n] if isinstance(gt[i][n],list) else [gt[i][n]]
            H=algo[i][n] if isinstance(algo[i][n],list) else [algo[i][n]]
            for l in L:
                for h in H:
                    df.append([i,n,l,h])
            
    df=pd.DataFrame(df)
    
    df.columns=['partite_id', 'nId', 'ground_truth_comm_id','assigned_comm_id']
    return df


#===============================================================================
# TEST - OK
#===============================================================================
if __name__=='__main__':
    he=[[1,2,3],
        [2,2,2],
        [4,5,6],
        [6,7,8]]
    
    comm1=[1,1,1,1,2,2,2,2,3,3,3,3]
    comm2=[12,12,12,12,23,23,23,23,34,34,34,34]
    
    d=to_dataframe(he, comm1, comm2)
    print(sp.array(he))
    print(comm1)
    print(comm2)
    print(d)