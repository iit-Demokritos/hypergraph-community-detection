'''
Created on Feb 22, 2016

@author: ak

source taken from:
http://stackoverflow.com/questions/8730119/retrieving-json-objects-from-a-text-file-using-python

'''

import json
from json.decoder import WHITESPACE

class ConcatJSONDecoder(json.JSONDecoder):
    """Custom json decoder for multiple json objects in one stream
    
    """
    def decode(self, s, _w=WHITESPACE.match):
        s_len = len(s)

        objs = []
        end = 0
        while end != s_len:
            obj, end = self.raw_decode(s, idx=_w(s, end).end())
            end = _w(s, end).end()
            objs.append(obj)
        return objs