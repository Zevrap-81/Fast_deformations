import os.path as osp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, BatchSampler
from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader

from trainer import TrainerBase
from trainer import load_checkpoint

from bodymodel import BodyModel
from deformer_model import SkinningModel
from parameters import Parameters

from dataset import DFaustDataset, Data
from sampler import RandomBatchSampler
from torch.utils.data import DataLoader

from utils import collate_fn

class Trainer(TrainerBase):
    def __init__(self, params: Parameters, **kwargs) -> None:
        super().__init__(params, **kwargs)
    
    def dataloader(self, dataset):
        batch_size= self.params.hyper.batch_size

        dataloader_ = DataLoader(dataset, batch_size=self.params.hyper.batch_size,
                                 shuffle=False, pin_memory=True,
                                 num_workers=3)
        return dataloader_

    def loss_config(self):
        mse = nn.MSELoss()
        def criterion(y_pred, y):
            loss = 1000*mse(y_pred, y)
            return loss
        self.criterion = criterion

    def train_step(self, data:Data, posedirs=None, **kwargs):
        data= data.to(self.params.hyper.device)
        # posedirs= posedirs.to(self.params.hyper.device)

        pred_deformations = self.model(data.transformations, posedirs)
        loss = self.criterion(pred_deformations, data.target_deformations)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_step(self, data:Data, posedirs=None, **kwargs):
        data= data.to(self.params.hyper.device)
        # posedirs= posedirs.to(self.params.hyper.device)

        pred_deformations = self.model(data.transformations, posedirs)
        loss = self.criterion(pred_deformations, data.target_deformations)
        return loss.item()

    def test(**kwargs):
        raise NotImplementedError(' Please use deformer model for inference!')

if __name__ == "__main__":
    params= Parameters()
    params.data.gender= 'male'
    params.hyper.num_epochs= 500
    params.hyper.batch_size= 1000
    params.hyper.lr= 1e-1
    dataset= DFaustDataset(params.data)
    trainset, valset= random_split(dataset, [params.data.split_ratio, 1-params.data.split_ratio], 
                                   generator=torch.Generator().manual_seed(params.data.random_seed))

    path= osp.join(params.data.data_dir, "DFaust_processed", 
            params.data.gender, rf"pca_per_{params.data.num_bones}_bones.pt")
    pcas_per_bone= torch.load(path)

    _, model_ckpt, trainer_ckpt= load_checkpoint(rf"saved_data\12_21_17_32\checkpoints\m-SkinningModel_e-120_ts-1000_l-6.79.pt")

    model= SkinningModel(params, pcas_per_bone)
    model.load_state_dict(model_ckpt)
    bm= BodyModel(params.data.bm_path, params.data.num_betas, params.data.gender)
    posedirs= bm.posedirs.to(params.hyper.device)

    # trainer= Trainer(params, model=model)
    trainer= Trainer.load_from_checkpoint(trainer_ckpt, params=params, model=model)
    trainer.train(trainset, valset, posedirs=posedirs)

    # #do inference
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