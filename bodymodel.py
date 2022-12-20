from typing import Optional, Dict
import os.path as osp 

import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F

from smplx.lbs import batch_rigid_transform, batch_rodrigues, vertices2joints 

from smplx.utils import (Struct, to_np, to_tensor, Tensor)

from smplx.vertex_ids import vertex_ids as VERTEX_IDS

from utils import Data


class BodyModel(nn.Module):

    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    SHAPE_SPACE_DIM = 300

    def __init__(
        self, model_path: str,
        num_betas: int = 10,
        dtype=torch.float32,
        gender: str = 'neutral',
        vertex_ids: Dict[str, int] = None,
        batch_size: int = None,
        **kwargs
    ) -> None:
        ''' SMPL model constructor '''

        self.gender = gender

        assert osp.exists(model_path), 'Path {} does not exist!'.format(model_path)
        data_struct = Struct(**np.load(model_path, 'rb', encoding='latin1', allow_pickle=True))

        super(BodyModel, self).__init__()
        
        shapedirs = data_struct.shapedirs
        if (shapedirs.shape[-1] < self.SHAPE_SPACE_DIM):
            print(f'WARNING: You are using a {self.name()} model, with only'
                  f' {shapedirs.shape[-1]} shape coefficients.\n'
                  f'num_betas={num_betas}, shapedirs.shape={shapedirs.shape}, '
                  f'self.SHAPE_SPACE_DIM={self.SHAPE_SPACE_DIM}')
            num_betas = min(num_betas, shapedirs.shape[-1])
        else:
            num_betas = min(num_betas, self.SHAPE_SPACE_DIM)
        self._num_betas = num_betas

        self.shapedirs = to_tensor(to_np(shapedirs[:, :, :num_betas]), dtype=dtype)

        if vertex_ids is None:
            # SMPL and SMPL-H share the same topology, so any extra joints can
            # be drawn from the same place
            vertex_ids = VERTEX_IDS['smplh']

        self.dtype = dtype

        self.faces =  to_tensor(to_np(data_struct.f, dtype=np.int64),
                                       dtype=torch.long)

        self.v_template = to_tensor(to_np(data_struct.v_template), dtype=dtype)

        self.J_regressor = to_tensor(to_np(data_struct.J_regressor), dtype=dtype)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data_struct.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.posedirs = to_tensor(to_np(posedirs), dtype=dtype)

        # indices of parents for each joints
        self.parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        self.parents[0] = -1

        self.lbs_weights = to_tensor(to_np(data_struct.weights), dtype=dtype)

        if not batch_size is None:
            self.init_defaults(batch_size)

    def init_defaults(self, batch_size):
        #todo
        self.batch_size= batch_size
        self.global_orient= None
        self.body_pose= None 

        self.betas = torch.zeros([1, self.num_betas], dtype=self.dtype)

    def forward_shape(
        self, betas=None, batch_size=1):
        # setting batch_size=1 as default for shape params 
        # change this setting 
        if betas is None:
            betas= self.betas
        else:
            if not torch.is_tensor(betas):
                betas = torch.tensor(betas, dtype=self.dtype)
        
        if betas.shape[0] != batch_size:
                betas= betas.expand(batch_size, -1)
        # self.register_parameter('betas', nn.Parameter(betas, requires_grad=True))
        v_shaped = self.v_template + torch.einsum('bl,mkl->bmk', [betas, self.shapedirs])
        J= vertices2joints(self.J_regressor, v_shaped)
        return v_shaped, J


    def forward_pose(self, 
        body_pose: Optional[Tensor],
        global_orient: Optional[Tensor],
        blend_pose: bool = False,
        batch_size: int= 1,
        v_shaped: Optional[Tensor]= None,
        J: Optional[Tensor]= None):

        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose

        full_pose = torch.cat([global_orient, body_pose], dim=1)


        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view([batch_size, -1, 3, 3])

        J_transformed, T = batch_rigid_transform(rot_mats, J, self.parents, dtype=self.dtype)
        
        v_posed= v_shaped

        if blend_pose:
            ident = torch.eye(3, dtype=self.dtype, device=rot_mats.device)
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
            # pose_feature [B, J * 9]
            # (N x P) x (P, V * 3) -> N x V x 3
            pose_offsets = torch.matmul(pose_feature, self.posedirs).view(batch_size, -1, 3)
            v_posed+= pose_offsets
        
        return v_posed, J_transformed, T 

    def forward_skinning(self, T, v_posed, W=None, batch_size=1):
        if W is None:
            W= self.lbs_weights
            W= W.unsqueeze(0).expand(batch_size, -1, -1)
        return lbs(W, T, v_posed)
         

    def forward(
        self,
        betas: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        trans: Optional[Tensor] = None,
        blend_pose: bool = False, 
        weights: Optional[Tensor] = None,
        batch_size: int = 1,
        **kwargs
    ):

        # If no shape and pose parameters are passed along, then use the
        # ones from the module

        v_shaped, joints= self.forward_shape(betas, batch_size)
        v_posed, joints, T= self.forward_pose(body_pose, global_orient, blend_pose, batch_size, v_shaped, joints)
        vertices= self.forward_skinning(T, v_posed, weights)

        if trans is not None:
            joints += trans.unsqueeze(dim=1)
            vertices += trans.unsqueeze(dim=1)
            T[..., -1][..., 0:3]+= trans.unsqueeze(dim=1)

        output = Data(  vertices=vertices,
                        global_orient=global_orient,
                        body_pose=body_pose,
                        joints=joints,
                        betas=betas,
                        transformations=T)

        return output


    @property
    def num_betas(self):
        return self._num_betas

    @property
    def num_expression_coeffs(self):
        return 0

    def name(self) -> str:
        return 'SMPL'

    @torch.no_grad()
    def reset_params(self, **params_dict) -> None:
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                param[:] = torch.tensor(params_dict[param_name])
            else:
                param.fill_(0)

    def get_num_verts(self) -> int:
        return self.v_template.shape[0]

    def get_num_faces(self) -> int:
        return self.faces.shape[0]

    def extra_repr(self) -> str:
        msg = [
            f'Gender: {self.gender.upper()}',
            f'Number of joints: {self.J_regressor.shape[0]}',
            f'Betas: {self.num_betas}',
            ]
        return '\n'.join(msg)


def lbs(W, T, v):
    v_homo= F.pad(v, (0, 1), value=1.0)
    v_skinned= torch.einsum("bpn,bnij,bpj->bpi", W, T, v_homo)
    return v_skinned[:, :, 0:3]

if __name__ == "__main__":
    model_path= r"C:\Users\Affu_rani\Downloads\3d_human\models_smplx_v1_1\smplx\models\smplx\SMPLX_FEMALE.npz"
    model= BodyModel(model_path, gender='male')
    
    bdata= np.load(r"C:\Users\Affu_rani\Downloads\3d_human\Dataset\DFaust\50002\50002_hips_stageii.npz")

    i=0
    b=len(bdata['poses'])
    betas, body_pose= torch.Tensor(bdata['betas'][:10]).unsqueeze(0), torch.Tensor(bdata['poses'][i:i+b, 3:]).reshape(b, -1, 3)
    global_orient= torch.Tensor(bdata['poses'][i:i+b, :3]).reshape(b, 1, 3)
    
    model.betas= betas
    output= model( body_pose=body_pose, global_orient=global_orient, batch_size=b, blend_pose=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    faces= model.faces
    from visualize import visualize_with_open3d
    visualize_with_open3d(vertices, faces)
    