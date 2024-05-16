import os
import hydra
import json

import pydicom as dcm
import SimpleITK as sitk
import numpy as np
from omegaconf import DictConfig
from loguru import logger
from tqdm import tqdm

from version import version_file_str
from segmentation.predict import Predict
from segmentation.segment import mask_to_contours


@hydra.main(version_base=None, config_path='..', config_name='config')
def segment_files(config: DictConfig) -> None:
    input_dir = config.segmentation.input_dir
    files = os.listdir(input_dir)
    predictor = Predict(main_window=None, config=config)

    for file in tqdm(files, desc='Segmenting files', unit='files', leave=False):
        try:
            image = dcm.read_file(os.path.join(input_dir, file), force=True).pixel_array
            if image.ndim == 4:  # 3 channel input
                image = image[:, :, :, 0]
        except (AttributeError, IsADirectoryError):
            try:  # NIfTi
                input_image = sitk.ReadImage(os.path.join(input_dir, file))
                image = sitk.GetArrayFromImage(input_image)
            except:
                logger.info(f'Skipping file {file} as it is not a valid IVUS file (DICOM or NIfTi supported)')
                continue

        logger.info(f'Segmenting file {file}')
        lower_limit = 0
        upper_limit = image.shape[0]
        masks = predictor(image, lower_limit, upper_limit)
        contours = mask_to_contours(None, masks, lower_limit, upper_limit, config=config)

        data = {}
        for key in [
            'lumen_area',
            'lumen_circumf',
            'longest_distance',
            'shortest_distance',
            'elliptic_ratio',
            'vector_length',
            'vector_angle',
        ]:
            data[key] = [0] * image.shape[0]
        for key in ['lumen_centroid', 'farthest_point', 'nearest_point']:
            data[key] = (
                [[] for _ in range(image.shape[0])],
                [[] for _ in range(image.shape[0])],
            )
        data['lumen'] = contours
        data['phases'] = ['-'] * image.shape[0]
        data['measures'] = [[None, None] for _ in range(image.shape[0])]
        data['measure_lengths'] = [[np.nan, np.nan] for _ in range(image.shape[0])]

        with open(os.path.join(input_dir, f'{file}_contours_{version_file_str}.json'), 'w') as out_file:
            json.dump(data, out_file)


if __name__ == '__main__':
    segment_files()
