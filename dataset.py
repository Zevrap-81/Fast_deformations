import os.path as osp
import glob

import numpy as np 
import torch
from torch.utils.data import Dataset
from trainer import Normalizer

from bodymodel import BodyModel
    
class PrepareBodyModel:
    def __init__(self, bm:BodyModel, num_bones) -> None:
        vertices_per_bone= self.get_vertices_per_bone(bm.lbs_weights)[:num_bones]
        self.pca_per_bone= self.get_pcas_per_bone(vertices_per_bone, bm.posedirs)

    def get_vertices_per_bone(self, weights):
        # get vertices influenced by bones i.e vertices with non-zero weights
        return [ torch.nonzero(t) for t in weights.T]
    
    def get_pcas_per_bone(self, vertices_per_bone, posedirs):
        # given a set of vertex ids find the corresponding pcas
        pca_per_bone= []
        for vertex_ids_per_bone in vertices_per_bone:
            vertices= np.zeros((10475, 3))
            vertices[vertex_ids_per_bone]= 100
            pca= np.dot(posedirs.reshape(-1, 486).T, vertices.flatten()) 
            pca_ids_per_bone= np.nonzero(pca)[0]
            pca_per_bone.append(pca_ids_per_bone)
        return pca_per_bone 


class DatasetBase(Dataset):
    def __init__(self, params) -> None:
        super().__init__() 
        self.params= params 

class DFaust_(DatasetBase):
    def __init__(self, params) -> None:
        super().__init__(params)
        pass 
    
    @torch.no_grad()
    def process(self):
        gender= self.params.gender
        bm_path= osp.join(self.params.smpl_dir, rf"SMPLX_{gender.upper()}.npz")
        bm= BodyModel(bm_path, num_betas=10, gender='male')
        
        num_bones= 24 #only considering smpl model bones
        bm_prep= PrepareBodyModel(bm, num_bones)
        torch.save(bm_prep.pca_per_bone, osp.join(self.processed_dir, r"pca_per_bone.pt"))
        
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
                    torch.save({"target_deformations": target_deformations, 
                                "transformations": transformations,
                                "betas":betas}, file_name)
                    sample+=1

                
    def get(self, idx):
        dir= osp.join(self.processed_dir, rf"data_{str(idx)}.pt")
        return torch.load(dir)

    def len(self):
        return len(glob.glob(osp.join(self.processed_dir, r"data_*.pt") ))

    def file_names(self):
        pass 

    @property
    def processed_dir(self):
        return osp.join(self.params.data_dir, "processed", rf"{self.params.gender}")
