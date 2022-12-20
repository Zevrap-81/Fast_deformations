
# from collections import namedtuple
# Data= namedtuple('Data', ['transformations', 'deformations', 'betas'])

# def to(self, device) -> Data:
#     return self.__class__(*[getattr(self, s).to(device) for s in self._fields])

#     # return Data(self.transformations.to(device), self.deformations.to(device), 
#     #             self.betas.to(device))
# Data.to= to

from torch_geometric.data import Dataset, Data as Data_
class Data(Data_):  
    'To force classical batching'
    def __cat_dim__(self, *args, **kwargs):
        return None