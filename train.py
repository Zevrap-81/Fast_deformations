import os.path as osp
from trainer import TrainerBase
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from bodymodel import BodyModel
from deformer_model import SkinningModel
from parameters import Parameters

from dataset import DFaustDataset, Data

class Trainer(TrainerBase):
    def __init__(self, params: Parameters, **kwargs) -> None:
        super().__init__(params, **kwargs)
    
    def dataloader(self, dataset):
        dataloader_ = DataLoader(dataset, batch_size=self.params.hyper.batch_size,
                                 shuffle=True, pin_memory=True,
                                 num_workers=4, persistent_workers=True)
        return dataloader_

    def loss_config(self):
        mse = nn.MSELoss()
        def criterion(y_pred, y):
            loss = 1000*mse(y_pred, y)
            return loss
        self.criterion = criterion

    def train_step(self, data:Data, posedirs=None, **kwargs):
        transformations= data.transformations.to(self.params.hyper.device)
        target_deformations= data.target_deformations.to(self.params.hyper.device)
        posedirs= posedirs.to(self.params.hyper.device)

        pred_deformations = self.model(transformations, posedirs)
        loss = self.criterion(pred_deformations, target_deformations)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_step(self, data:Data, posedirs=None, **kwargs):
        transformations= data.transformations.to(self.params.hyper.device)
        target_deformations= data.target_deformations.to(self.params.hyper.device)
        posedirs= posedirs.to(self.params.hyper.device)

        pred_deformations = self.model(transformations, posedirs)
        loss = self.criterion(pred_deformations, target_deformations)
        return loss.item()

    def test(**kwargs):
        raise NotImplementedError(' Please use deformer model for inference!')

if __name__ == "__main__":
    params= Parameters()
    params.data.gender= 'female'
    params.hyper.num_epochs= 500
    params.hyper.batch_size= 2000
    dataset= DFaustDataset(params.data)
    exit()
    trainset, valset= random_split(dataset, [params.data.split_ratio, 1-params.data.split_ratio], 
                                   generator=torch.Generator().manual_seed(params.data.random_seed))

    path= osp.join(params.data.data_dir, "DFaust_processed", 
            params.data.gender, rf"pca_per_{params.data.num_bones}_bones.pt")
    pcas_per_bone= torch.load(path)


    model= SkinningModel(params, pcas_per_bone)
    bm= BodyModel(params.data.bm_path, params.data.num_betas, params.data.gender)

    trainer= Trainer(params, model=model)
    trainer.train(trainset, valset, posedirs=bm.posedirs)

    # #do inference
    # from trainer import load_checkpoint
    # from deformer_model import Deformer
    # from torch_geometric.loader import DataLoader
    # from visualize import visualize_with_open3d

    # _, model_ckpt, _= load_checkpoint(rf"saved_data\12_21_14_57\checkpoints\m-SkinningModel_e-5_ts-1000_l-1.35.pt")
    # deformer= Deformer(params=params, model_ckpt=model_ckpt)

    # dataloader= DataLoader(valset, params.hyper.batch_size)

    # for data in dataloader:
    #     betas, poses= data.betas, data.poses 
    #     global_orient, body_pose= poses[:, :3], poses[:, 3:]

    #     output= deformer(betas=betas, global_orient=global_orient, body_pose=body_pose, batch_size=poses.size(0))
    #     vertices = output.vertices.detach().cpu().numpy().squeeze()
    #     faces= deformer.faces
    #     visualize_with_open3d(vertices, faces)
    #     break