import os.path as osp 
from collections import OrderedDict
from typing import Optional

import torch 
import torch.nn as nn 

from bodymodel import BodyModel
from parameters import Parameters, HyperParameters

class Model_per_Bone(nn.Module):
    def __init__(self, params:HyperParameters, in_channel, out_channel) -> None:
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

        
class SkinningModel(nn.Sequential):

    def __init__(self, params:Parameters, pcas_per_bone) -> None:
        self.params= params
        self.pcas_per_bone= pcas_per_bone
        models= [Model_per_Bone(params.hyper, 16, len(a)) for a in pcas_per_bone]
        super().__init__(*models) 
        
    def forward(self, transformations, posedirs):
        batch_size= transformations.size(0)
        kwargs= dict( device=transformations.device, dtype=transformations.dtype)
        output= torch.zeros((batch_size, self.params.data.num_tethas), **kwargs)
        
        #todo use multiprocessing to run all models in parallel
        for i in range(self.params.data.num_bones):
            model_i= self[i]
            T_i= transformations[:, i]
            pca_ids= self.pcas_per_bone[i]
            output[:,pca_ids]+= model_i(T_i)
            
        output= torch.matmul(output, posedirs).view(batch_size, -1, 3)
        return output
    

class Deformer(BodyModel):
    # Do inference
    input_norm, output_norm= None, None

    def __init__(self, params=None, model_ckpt=None) -> None:
        self.params= params
        super().__init__(self.params.data.bm_path, num_betas=self.params.data.num_betas, gender=self.params.data.gender)

        path= osp.join(self.params.data.data_dir, "DFaust_processed", 
              self.params.data.gender, rf"pca_per_{self.params.data.num_bones}_bones.pt")
        pcas_per_bone= torch.load(path)

        if model_ckpt:
            self.model= SkinningModel(self.params, pcas_per_bone)
            self.model.load_state_dict(model_ckpt)
        else:
            raise Exception("Please provide model checkpoint")

        self.model.to(self.params.hyper.device)

    def forward_skinning(self, T, v_posed, W=None, batch_size=1):

        l_disp= super().forward_skinning(T, v_posed, W, batch_size)

        if self.input_norm is None or self.output_norm is None:
            norms= torch.load(osp.join(self.params.data.data_dir, "DFaust_processed", 
                              self.params.data.gender, r'norms.pt'))
            self.input_norm= norms['input_norm']
            self.output_norm= norms['output_norm']

        transformations= T[:, :self.params.data.num_bones].view(-1, self.params.data.num_bones, 16)
        transformations= self.input_norm(transformations).to(self.params.hyper.device)
        posedirs= self.posedirs.to(self.params.hyper.device)
        non_l_disp= self.model(transformations, posedirs)
        non_l_disp= self.output_norm(non_l_disp, inverse=True).cpu()
        
        disp= l_disp + non_l_disp       #additive model
        return disp


    def forward(
        self,
        betas: Optional[torch.Tensor] = None,
        body_pose: Optional[torch.Tensor] = None,
        global_orient: Optional[torch.Tensor] = None,
        trans: Optional[torch.Tensor] = None,
        blend_pose: bool = False, 
        weights: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        **kwargs
    ):

        # manually set blend_pose to false
        return super().forward(blend_pose=False, betas=betas, body_pose=body_pose, global_orient=global_orient, trans=trans, weights=weights,
                               batch_size=batch_size, **kwargs)
