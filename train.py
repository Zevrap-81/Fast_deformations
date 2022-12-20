from trainer import TrainerBase
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from parameters import Parameters

from dataset import Data

class Trainer(TrainerBase):
    def __init__(self, params: Parameters, **kwargs) -> None:
        super().__init__(params, **kwargs)
    
    def dataloader(self, dataset):
        dataloader_ = DataLoader(dataset, batch_size=self.params.hyper.batch_size,
                                 shuffle=True, pin_memory=True,
                                 num_workers=1, persistent_workers=True)
        return dataloader_

    def loss_config(self):
        mse = nn.MSELoss()
        def criterion(y_pred, y):
            loss = 100*mse(y_pred, y)
            return loss
        self.criterion = criterion

    def train_step(self, data:Data, posedirs=None, **kwargs):
        data = data.to(self.params.hyper.device)
        prediction = self.model(data.transformations, posedirs)
        loss = self.criterion(prediction, data.deformations)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_step(self, data:Data, posedirs=None, **kwargs):
        data=  data.to(self.params.hyper.device)
        prediction= self.model(data.transformations, posedirs)
        loss= self.criterion(prediction, data.deformations)
        return loss.item()

    def test(**kwargs):
        raise NotImplementedError(' Please use deformer model for inference!')
