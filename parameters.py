from dataclasses import dataclass
from trainer import Parameters as ParamsBase, HyperParameters as HyperParamsBase, DataParameters as DataParamsBase

@dataclass
class HyperParameters(HyperParamsBase):
    num_layers:int= 2 

@dataclass
class DataParameters(DataParamsBase):
    gender:str= 'male'
    smpl_dir:str= r"C:\Users\Affu_rani\Downloads\3d_human\models_smplx_v1_1\smplx\models\smplx"
    data_dir:str= r"C:\Users\Affu_rani\Downloads\3d_human\Dataset\DFaust"
    skip:int= 2
    
class Parameters(ParamsBase):
    def __init__(self,  data=DataParameters(), 
                        hyper=HyperParameters()) -> None:
        super().__init__(data, hyper)