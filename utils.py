# from torch_geometric.data import Data as Data_
# class Data(Data_):  
#     'To force classical batching'
#     def __cat_dim__(self, *args, **kwargs):
#         return 0

import torch
class Data(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)
    
    @property
    def _fields(self):
        return self.__dict__.keys()

    def __str__(self):
        return 'Data {' + ', '.join([f'{key}:{list(getattr(self, key).shape)}' for key in self._fields]) + '}'

    def to(self, device):
        for key in self._fields:
            setattr(self, key, getattr(self, key).to(device))
        return self

def collate_fn(data):
    elem= data[0]
    keys= elem._fields

    out= {}
    for key in keys:
        out[key]= torch.cat([getattr(d, key) for d in data])
    return elem.__class__(**out)