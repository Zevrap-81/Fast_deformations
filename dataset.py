import os.path as osp
import glob

import numpy as np 
import torch
from trainer import Normalizer

from bodymodel import BodyModel
from parameters import DataParameters

class PrepareBodyModel:
    def __init__(self, bm:BodyModel, num_bones) -> None:
        vertices_per_bone= self.get_vertices_per_bone(bm.lbs_weights)[:num_bones]
        self.pca_per_bone= self.get_pcas_per_bone(vertices_per_bone, bm.posedirs)

    def get_vertices_per_bone(self, weights):
        # get vertices influenced by bones i.e vertices with non-zero weights
        return [ torch.nonzero(t) for t in weights.T]
    
    def get_pcas_per_bone(self, vertices_per_bone, posedirs):
        # given a set of vertex ids find the corresponding pcas
        pcas_per_bone= []
        for vertex_ids_per_bone in vertices_per_bone:
            vertices= np.zeros((10475, 3))
            vertices[vertex_ids_per_bone]= 100
            pca= np.dot(posedirs.reshape(-1, 486).T, vertices.flatten()) 
            pca_ids_per_bone= np.nonzero(pca)[0]
            pcas_per_bone.append(pca_ids_per_bone)
        return pcas_per_bone 


from torch_geometric.data import Dataset
from utils import Data

#todo check the requirements of the torchgeometric dataset class 
class DFaust_(Dataset):
    def __init__(self, params:DataParameters) -> None:
        self.params= params
        super().__init__(params.data_dir)
         
    
    @torch.no_grad()
    def process(self):
        gender= self.params.gender
        bm= BodyModel(self.params.bm_path, num_betas=self.params.num_betas, gender=gender)
        
        num_bones= self.params.num_bones # 24 only considering smpl model bones
        bm_prep= PrepareBodyModel(bm, num_bones)
        torch.save(bm_prep.pca_per_bone, osp.join(self.processed_dir, rf"pca_per_{num_bones}_bones.pt"))
        
        subjects = [osp.basename(s_dir) for s_dir in sorted(glob.glob(osp.join(self.params.data_dir, '*')))]

        input_norm= Normalizer('transformations')
        output_norm= Normalizer('deformations')

        sample= 0
        for subject in subjects:
            subject_dir= osp.join(self.params.data_dir, subject)

            shape_dir= osp.join(subject_dir, rf'{gender}_stagei.npz')
            if not osp.isfile(shape_dir):
                # do only the specified gender files
                continue

            betas= torch.from_numpy(np.load(shape_dir, allow_pickle=True)['betas'])
            bm.betas= betas.unsqueeze(0)
            # save_dir= osp.join(self.processed_dir, rf"{subject}")
            
            for s_dir in glob.glob(osp.join(subject_dir, r'{subject}_*.npz')):
                subject_data= np.load(s_dir, allow_pickle=True)
                poses= subject_data['poses'][::self.params.skip]
                global_orient= torch.from_numpy(poses[:, :3])
                body_pose= torch.from_numpy(poses[:, 3:])
                trans= torch.from_numpy(subject_data['trans'][::self.params.skip])

                batch_size= body_pose.size(0)
                body= bm(blend_pose=True, global_orient=global_orient, body_pose=body_pose, trans=trans,  batch_size=batch_size)

                transformations= body.transformations[:, :num_bones, :, :].view(-1, num_bones, 16)
                #todo add them to cpu
                input_norm.fit(transformations)

                #todo get linear deformations
                linear_body= bm(blend_pose=False, global_orient=global_orient, body_pose=body_pose, trans=trans,  batch_size=batch_size)

                target_deformations= body.vertices - linear_body.vertices
                output_norm.fit(target_deformations)

                for i in range(batch_size):

                    file_name= osp.join(self.processed_dir, rf"data_{sample}.pt")
                    torch.save(Data(transformations=transformations, target_deformations=target_deformations, betas=betas), file_name)
                    sample+=1

        torch.save({'input_norm': input_norm,
                    'output_norm': output_norm}, 
                    osp.join(self.processed_dir, r"norms.pt"))
                
    def get(self, idx):
        dir= osp.join(self.processed_dir, rf"data_{str(idx)}.pt")
        return torch.load(dir)

    def len(self):
        return len(glob.glob(osp.join(self.processed_dir, r"data_*.pt") ))

    def file_names(self):
        pass 

    @property
    def processed_dir(self):
        return osp.join(self.params.data_dir, "processed", self.params.gender)
