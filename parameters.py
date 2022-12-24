import os.path as osp

from dataclasses import dataclass
from trainer import Parameters as ParamsBase, HyperParameters as HyperParamsBase, DataParameters as DataParamsBase

@dataclass
class HyperParameters(HyperParamsBase):
    num_layers:int= 2 

@dataclass
class DataParameters(DataParamsBase):
    gender:str= 'male'
    num_betas:int= 10
    num_tethas:int= 486
    num_bones:int= 24
    smpl_dir:str= r"C:\Users\Affu_rani\Downloads\3d_human\models_smplx_v1_1\smplx\models\smplx"
    data_dir:str= r"C:\Users\Affu_rani\Downloads\fast_deformations\data"
    skip:int= 3

    @property
    def bm_path(self):
        return osp.join(self.smpl_dir, rf"SMPLX_{self.gender.upper()}.npz")
    
class Parameters(ParamsBase):
    def __init__(self,  data:DataParameters=DataParameters(), 
                        hyper:HyperParameters=HyperParameters()) -> None:
        super().__init__(data, hyper)