import sys
import torch
from ..async_tile_processor import async_tile_processor
import numpy as np
import torch.nn as nn
import sys
import yaml
from torchvision import models

class MIL_processor(async_tile_processor): 

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.softmax_fn = nn.Softmax()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_config_from_yaml(self, config_path: str) -> dict:
        """_summary_

        Args:
            config_path (str): _description_

        Returns:
            dict: _description_
        """
        with open(file=config_path, mode="r") as param_file:
            parameters = yaml.load(stream=param_file, Loader=yaml.SafeLoader)
        return parameters["model"], parameters["sampler"], parameters["training"]

    # def _load_network_model(self):
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #     model = UNet(
    #         3,
    #         64,
    #         "batch_norm",
    #         0.2,
    #     )
    #     model.load_state_dict(torch.load(self._model_path))

    #     model.to(device) 
    #     model.eval()
        
    #     return model   
            
    def _load_network_model(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        config_path = self._model_path.split("model_")[0] + "model.yaml"

        model_parameters, _, _ = self.get_config_from_yaml(config_path)
        self.nclasses = int(model_parameters['nclasses'])
        
        if model_parameters['backbone'] == "resnet50":
            model = models.resnet50()
            model.fc = nn.Linear(2048, self.nclasses)
        else:
            model = models.efficientnet_b4()

        state_dict = torch.load(self._model_path)
        model.load_state_dict(state_dict)
        print("Model succesfully loaded!")
        model.to(self.device)

        return model

    def _predict_tile_batch(self, tile_batch=None, info=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self._ax_order == 'cwh':
            tile_batch = tile_batch.transpose(0, 3, 1, 2)
        
        tile_batch = torch.from_numpy(tile_batch).to(device)

        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                result = self._model(tile_batch)
        
        result = self.softmax_fn(result)
        result = result.detach().cpu().numpy()
        result = np.expand_dims(np.expand_dims(result, axis=-1), axis=-1)

        result_shape = list(tile_batch.shape)
        result_shape[1] = self.nclasses
        tile_batch = np.ones(result_shape)
        result = tile_batch * result

        if self._ax_order == 'cwh':
            result = result.transpose(0, 2, 3, 1)
        
        return result.astype(np.float32)

    def _send_reconstruction_info(self):
        self._write_queues[0].put(('recon_info','',1))