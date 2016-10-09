'''
Created on Mar 11, 2016

@author: ak
'''
import json
from Utils.ConcatJSONDecoder import ConcatJSONDecoder

def json_reader(filename, json_list=False,int_keys=False):
    """read json file to dictionary
    
    Parameters
    ----------
    filename: str
        json file path
    json_list: bool, optional
        list=True if json contains list of json objects
        default=False
    int_keys: bool, optional
        if dict and keys must be int, convert them (json accepts only string keys)
    Returns
    -------
    json_data: dict
        dictionary of parsed json data
    """
    
    #DBG
    print('...reading json...')
    json_data=None
    with open(filename) as json_file:
        if json_list:
            json_data=json.load(json_file, cls=ConcatJSONDecoder)
        else:
            json_data=json.load(json_file)
    
    if isinstance(json_data, dict) and int_keys:
        json_data={int(k):json_data[k] for k in json_data}
    
    if isinstance(json_data, list) and int_keys:
        if isinstance(json_data[0], dict):
            json_data=[{int(k):d[k] for k in d} for d in json_data]
            
    return json_data

def json_writer(obj, filename):
    """write obj data (assume well formatted) to json file
    
    Parameters
    ----------
    obj: object to dump to json
    filename: string
        filename of json file (with extension .json)
    
    """
    with open(filename, 'w') as f:
        json.dump(obj, f)
