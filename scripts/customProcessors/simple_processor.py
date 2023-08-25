import sys
from ..async_tile_processor import async_tile_processor
import numpy as np

import sys
sys.path.insert(1, './pathology-NASH-quantification')
from scripts.previousThesis import HNE_segmentation, PSR_segmentaiton

class simple_processor(async_tile_processor): 

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  

    def _load_network_model(self):
        self._level = int(self._model_path.split('_')[1])
        self._dsl = int(self._model_path.split('_')[2])
        self._model_path = self._model_path.split('_')[0]

        if self._model_path == 'HNE':
            model = HNE_segmentation.segment_both
        else:
            model = PSR_segmentaiton.fibrosis_segmentation

        if self._model_path == "PSR":
            self.base_shape = 2
        else:
            self.base_shape = 3

        return model           

    def _predict_tile_batch(self, tile_batch=None, info=None):

        shape = list(tile_batch.shape)
        shape[-1] = self.base_shape
        result = np.zeros(shape)
        print('start')
        for i, tile in enumerate(tile_batch):

            if self._model_path == 'HNE':
                tile_result = self._model(tile, level=self._level, dsl=self._dsl)
            else:
                tile_result = self._model(tile)

            if len(tile_result) == 2:
                tile_result = tile_result[0]

            for j in range(self.base_shape+1):
                result[i, tile_result == j, j] = 1

        print(result.shape)

        return result


    def _send_reconstruction_info(self):
        self._write_queues[0].put(('recon_info','',1))