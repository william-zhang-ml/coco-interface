"""
Class to interface with Microsoft COCO 2017 dataset.

Author: William Zhang
"""
import json
import os
from typing import Dict, List, Tuple
import torch
import torchvision


class CocoInstances:
    """ Interface for COCO 2017 detection/segmentation instances. """
    def __init__(self,
                 metadata_path: str,
                 img_dir: str) -> None:
        """Load and process metadata.

        Args:
            metadata_path (str): path to some form of instances_????.json
            img_dir (str): path to image directory
        """
        self.metadata_path = metadata_path
        self.img_dir = img_dir

        # load metadata for dataset split
        with open(metadata_path, 'r', encoding='utf-8') as file:
            self.metadata = json.load(file)

        # bookkeep how image IDs link to annotation indices
        self.img_id_to_annot_idcs: Dict[int, List[int]] = {
            img_meta['id']: [] for img_meta in self.metadata['images']
        }
        for annot_idx, annot in enumerate(self.metadata['annotations']):
            self.img_id_to_annot_idcs[annot['image_id']].append(annot_idx)

    def __len__(self) -> int:
        return len(self.metadata['images'])

    def __repr__(self) -> str:
        argstr = ', '.join([
            f'metadata_path={self.metadata_path}',
            f'img_dir={self.img_dir}'
        ])
        return f'CocoInstance({argstr})'

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[Dict]]:
        """Get a single image and its set of instance annotations.

        Args:
            index (int): dataset index

        Returns:
            Tuple[torch.Tensor, List[Dict]]: image, annotations
        """
        img = torchvision.io.read_image(
            os.path.join(
                self.img_dir,
                self.metadata['images'][index]['file_name']
            ),
            torchvision.io.ImageReadMode.RGB
        )
        curr_image_id = self.metadata['images'][index]['id']
        annotations = [
            self.metadata['annotations'][idx]
            for idx in self.img_id_to_annot_idcs[curr_image_id]
        ]  # TODO - give user a copy of this data, not the original refs
        return img, annotations
