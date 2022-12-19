from collections import OrderedDict
import torch.nn as nn 

class Model_per_Bone(nn.Module):
    def __init__(self, params, in_channel, out_channel) -> None:
        super().__init__()

        module_dict= OrderedDict()
        for l in range(params.num_layers):
            if l==0:
                module_dict['input_layer'] = nn.Linear(in_channel, params.hidden_channel)
                module_dict['input_act'] = params.act

            elif l == params.num_layers-1:
                module_dict['output_layer'] = nn.Linear(params.hidden_channel, out_channel)
            
            else:
                module_dict['layer_'+str(l)] = nn.Linear(params.hidden_channel, params.hidden_channel)
                module_dict['act_'+str(l)] = params.act

        self.model= nn.Sequential(module_dict) 

    def forward(self, T_j):
        predictions= self.model(T_j)
        return predictions
        
        # torch.einsum("bm,ndm->bnd", predictions, posedirs) #change this to bmm later


class Deformer(nn.Sequential):
    def __init__(self, params, num_bones, pcas_per_bone) -> None:
        super().__init__()
        pass 

    def forward(self):
        pass 

    def forward_skinning(self):
        pass 

