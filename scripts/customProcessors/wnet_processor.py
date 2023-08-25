import sys
import torch
from ..async_tile_processor import async_tile_processor
import numpy as np
import torch.nn as nn
import sys
sys.path.insert(1, './pathology-NASH-quantification')
from scripts.models.w_net2 import UNet

class wnet_processor(async_tile_processor): 

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.softmax_fn = nn.Softmax2d()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_network_model(self):

        model = UNet(
            3,
            64,
            "batch_norm",
            0.2,
        )
        model.load_state_dict(torch.load(self._model_path))

        model.to(self.device) 
        model.eval()
        
        return model   
            
    def _predict_tile_batch(self, tile_batch=None, info=None):
        if self._ax_order == 'cwh':
            tile_batch = tile_batch.transpose(0, 3, 1, 2)
        
        tile_batch = torch.from_numpy(tile_batch).to(self.device)
        
        with torch.cuda.amp.autocast():   
            result = self._model.predict(tile_batch)
        
        result = self.softmax_fn(result)
        result = result.detach().cpu().numpy()

        if self._ax_order == 'cwh':
            result = result.transpose(0, 2, 3, 1)
        
        return result

    def _send_reconstruction_info(self):
        self._write_queues[0].put(('recon_info','',1))