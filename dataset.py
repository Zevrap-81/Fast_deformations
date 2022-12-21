import os.path as osp
import glob

import numpy as np 
import torch
from trainer import Normalizer
from tqdm import tqdm

from bodymodel import BodyModel
from parameters import DataParameters

class PrepareBodyModel:
    def __init__(self, bm:BodyModel, num_bones) -> None:
        vertices_per_bone= self.get_vertices_per_bone(bm.lbs_weights)[:num_bones]
        self.pca_per_bone= self.get_pcas_per_bone(vertices_per_bone, bm.posedirs)

    def get_vertices_per_bone(self, weights):
        # # get vertices influenced by bones i.e vertices with non-zero weights
        return [ torch.nonzero(t).squeeze(1) for t in weights.T]
    
    def get_pcas_per_bone(self, vertices_per_bone, posedirs):
        # given a set of vertex ids find the corresponding pcas
        pcas_per_bone= []
        for vertex_ids_per_bone in vertices_per_bone:
            vertices= torch.zeros((10475, 3), device=posedirs.device)
            vertices[vertex_ids_per_bone]= 100
            pca= torch.matmul(posedirs, vertices.flatten()) 
            pca_ids_per_bone= torch.nonzero(pca).squeeze(1)
            pcas_per_bone.append(pca_ids_per_bone)
        return pcas_per_bone 


from torch_geometric.data import Dataset
from utils import Data

#todo check the requirements of the torchgeometric dataset class 
class DFaustDataset(Dataset):

    input_norm, output_norm= None, None

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

        subjects = [osp.basename(osp.normpath(s_dir)) 
                    for s_dir in sorted(glob.glob(osp.join(self.data_dir, '*', '')))]

        input_norm= Normalizer('transformations')
        output_norm= Normalizer('deformations')

        sample= 0
        for subject in tqdm(subjects, unit='subject', colour='red'):
            subject_dir= osp.join(self.data_dir, subject)

            shape_dir= osp.join(subject_dir, rf'{gender}_stagei.npz')
            if not osp.isfile(shape_dir):
                # do only the specified gender files
                continue

            betas= torch.Tensor(np.load(shape_dir, allow_pickle=True)['betas'][:self.params.num_betas])
            bm.betas= betas.unsqueeze(0)

            for s_dir in glob.glob(osp.join(subject_dir, rf'{subject}_*.npz')):
                subject_data= np.load(s_dir, allow_pickle=True)

                poses= torch.Tensor(subject_data['poses'][::self.params.skip])
                global_orient= poses[:, :3]
                body_pose= poses[:, 3:]

                trans= torch.Tensor(subject_data['trans'][::self.params.skip])

                batch_size= body_pose.size(0)
                body= bm(blend_pose=True, global_orient=global_orient, body_pose=body_pose, trans=trans,  batch_size=batch_size)

                transformations= body.transformations[:, :num_bones, :, :].view(-1, num_bones, 16)
                input_norm.fit(transformations)

                # get linear deformations
                linear_body= bm(blend_pose=False, global_orient=global_orient, body_pose=body_pose, trans=trans,  batch_size=batch_size)

                target_deformations= body.vertices - linear_body.vertices
                output_norm.fit(target_deformations)

                for i in range(batch_size):
                    file_name= osp.join(self.processed_dir, rf"data_{sample}.pt")
                    torch.save(Data(transformations=transformations[i], target_deformations=target_deformations[i], betas=betas, poses=poses[i]), file_name)
                    sample+=1

        torch.save({'input_norm': input_norm,
                    'output_norm': output_norm}, 
                    osp.join(self.processed_dir, r"norms.pt"))
                
    def get(self, idx):
        dir= osp.join(self.processed_dir, rf"data_{str(idx)}.pt")
        data= torch.load(dir)

        if self.input_norm is None or self.output_norm is None:
            norms= torch.load(osp.join(self.processed_dir, r'norms.pt'))
            self.input_norm= norms['input_norm']
            self.output_norm= norms['output_norm']
        
        data.transformations= self.input_norm(data.transformations)
        data.target_deformations= self.output_norm(data.target_deformations)
        return data

    def len(self):
        return len(glob.glob(osp.join(self.processed_dir, r"data_*.pt") ))

    def processed_file_names(self):
        return [r"data_00.pt", rf"data_{self.len()-1}.pt"] 

    @property
    def data_dir(self):
        return osp.join(self.params.data_dir, 'DFaust')

    @property
    def processed_dir(self):
        return osp.join(self.params.data_dir, "DFaust_processed", self.params.gender)


if __name__ == "__main__":
    data_params= DataParameters()
    dataset= DFaustDataset(data_params)
    print(dataset[0])

    ## Testing dataset class functionality
    # model_path= r"C:\Users\Affu_rani\Downloads\3d_human\models_smplx_v1_1\smplx\models\smplx\SMPLX_FEMALE.npz"
    # model= BodyModel(model_path, gender='male')
    
    # from torch_geometric.loader import DataLoader
    # dataloader= DataLoader(dataset, 1000, num_workers=3)

    # data= next(iter(dataloader))
    # poses= data.poses
    # betas= data.betas
    # body_pose= poses[:, 3:]
    # global_orient= poses[:, :3]
    # output= model(betas=betas, body_pose=body_pose, global_orient=global_orient, batch_size=1000, blend_pose=True)
    # vertices = output.vertices.detach().cpu().numpy().squeeze()
    # faces= model.faces
    # from visualize import visualize_with_open3d
    # visualize_with_open3d(vertices, faces)

