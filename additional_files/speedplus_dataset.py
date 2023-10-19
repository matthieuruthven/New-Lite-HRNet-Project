# Custom MMPose dataset class for the SPEED+ dataset
# Author: Matthieu Ruthven (matthieu.ruthven@uni.lu)
# Last modified: 11th October 2023

# Import required modules
import json
import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.fileio import exists, get_local_path

from mmpose.registry import DATASETS
from mmengine.dataset import BaseDataset


@DATASETS.register_module(name='SpeedPlusDataset')
class SpeedPlusDataset(BaseDataset):
    """SPEED+ Dataset for top-down pose estimation.

    "Next Generation Spacecraft Pose Estimation Dataset (SPEED+)"
    <https://zenodo.org/record/5588480>

    Args:
        ann_file (str): Annotation file path. Default: ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Default: ``None``.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Default: ``None``.
        data_prefix (dict, optional): Prefix for training data. Default:
            ``dict(img=None, ann=None)``.
        filter_cfg (dict, optional): Config for filter data. Default: `None`.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Default: ``None`` which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy.
            Default: ``True``.
        pipeline (list, optional): Processing pipeline. Default: [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Default: ``False``.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Default: ``False``.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Default: 1000.
    """

    METAINFO: dict = dict(from_file='configs/_base_/datasets/speedplus.py')

    def __init__(self,
                 ann_file: str = '',
                 data_mode: str = 'topdown',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        
        if data_mode not in {'topdown'}:
            raise ValueError(
                f'{self.__class__.__name__} got invalid data_mode: '
                f'{data_mode}. Should be "topdown".')
        self.data_mode = data_mode

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)
    
    @staticmethod
    def _is_valid_instance(data_info: Dict) -> bool:
        """Check JSON file."""
        # invalid keypoints
        if 'num_keypoints' in data_info and data_info['num_keypoints'] == 0:
            return False
        # invalid keypoints
        if 'keypoints' in data_info:
            if np.max(data_info['keypoints']) <= 0:
                return False
        return True

    def _get_topdown_data_infos(self, instance_list: List[Dict]) -> List[Dict]:
        """Organize the data list in top-down mode."""
        # Check labels
        data_list_tp = list(filter(self._is_valid_instance, instance_list))

        return data_list_tp
    
    def load_data_list(self) -> List[dict]:
        """Load data list from COCO annotation file or person detection result
        file."""

        instance_list, image_list = self._load_annotations()

        if self.data_mode == 'topdown':
            data_list = self._get_topdown_data_infos(instance_list)
        else:
            assert self.data_mode == 'topdown', 'The data mode should be "topdown"'

        return data_list
    
    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        """Load JSON file of keypoint coordinates and visibility indicators."""

        # Check that JSON file exists
        assert exists(self.ann_file), 'Annotation file does not exist'
        
        # Read JSON file
        with get_local_path(self.ann_file) as local_path:
            with open(local_path) as anno_file:
                self.anns = json.load(anno_file)

        # Preallocate lists 
        instance_list = []
        image_list = []
        used_img_ids = set()
        ann_id = 0

        # For each image
        for ann in self.anns:

            # Load keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
            keypoints = np.array(ann['keypoints']).reshape(1, -1, 3)
            keypoints = keypoints[..., :-1]
            keypoints_visible = np.array(ann['keypoints'][2::3]).reshape(1, -1)

            instance_info = {
                'id': ann_id,
                'img_id': int(ann['filename'][3:-4]),
                'img_path': osp.join(self.data_prefix['img'], ann['filename']),
                'input_size': (640, 640),
                'keypoints': keypoints,
                'keypoints_visible': keypoints_visible,
                'bbox': np.array([0, 0, 640, 640], dtype=np.float32).reshape(1, 4),
                'bbox_score': np.ones(1, dtype=np.float32)
            }

            if instance_info['img_id'] not in used_img_ids:
                used_img_ids.add(instance_info['img_id'])
                image_list.append({
                    'img_id': instance_info['img_id'],
                    'img_path': instance_info['img_path']
                })

            instance_list.append(instance_info)
            ann_id = ann_id + 1

        return instance_list, image_list