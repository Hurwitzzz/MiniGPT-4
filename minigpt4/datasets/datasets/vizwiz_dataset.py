"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json

from PIL import Image

from minigpt4.datasets.datasets.vqa_datasets import VQADataset

from collections import OrderedDict
import random

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "answer": ann["answers"],
                # "answers": "_".join(ann["answer"]),
                # "answers": '_'.join([answer['answer'] for answer in ann['answers']]),
                "image": sample["image"],
            }
        )


class VizwizDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        question = f"[vqa] The question is '{question}' Based on the image, answer the question with a single word or phrase. and reply 'unanswerable' when the provided information is insufficient"

        answers = ann['answers']
        answers = '_'.join([answer['answer'] for answer in answers])
        answers = self.text_processor(answers)

        return {
            "image": image,
            "question": question,
            "answer": answers,
        }

