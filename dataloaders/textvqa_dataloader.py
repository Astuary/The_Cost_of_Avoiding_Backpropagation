import os
import sys

import json
import copy
from typing import Dict, Sequence

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import pil_to_tensor, resize
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

from PIL import Image
import pandas as pd
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    PreTrainedTokenizer
)

class TextVQATrainDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def get_image(self, image):
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
            
        image_tensor = pil_to_tensor(image)
        image_tensor = resize(image_tensor, (512, 512), antialias=True)
        return image_tensor
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = self.get_image(item['image'])
        
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        
        if image is None:
            return None
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "index": 0,
                        "type": "image",
                        "text": "null",
                    },
                    {
                        "index": "null",
                        "type": "text",
                        "text": item["question"],
                    }
                ]
            },   
            {
                "role": "assistant",
                "content": [
                    {
                        "index": "null",
                        "type": "text",
                        "text": ";".join(item["answers"]),
                    }
                ]
            },   
        ]
        
        return {
            'image': image,
            'messages': messages,
        }

class TextVQATestDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def get_image(self, image):
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
    
        image_tensor = pil_to_tensor(image)
        image_tensor = resize(image_tensor, (512, 512), antialias=True)
        return image_tensor
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = self.get_image(item['image'])
        
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "index": 0,
                        "type": "image",
                        "text": "null",
                    },
                    {
                        "index": "null",
                        "type": "text",
                        "text": item["question"],
                    }
                ]
            },   
        ]
         
        return {
            'image': image,
            'messages': messages,
            'answers': ";".join(item['answers'])
        }

class Qwen2DataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        
        examples = [ex for ex in examples if ex is not None]
        
        for example in examples:
            messages = example["messages"]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
            images.append(example["image"])

        batch = self.processor(
            text=texts, 
            images=images, 
            return_tensors="pt", 
            padding=True
        )

        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        if "answers" in list(examples[0].keys()):
            answers = []
            for example in examples:
                answers.append(example["answers"])
            batch["answers"] = answers

        return batch
    

def get_centralized_dataset(
    processor,
    train_batch_size: int = 6,
    test_batch_size: int = 10,
    max_seq_len: int = 100,
    random_seed: int = 0,
    cache_dir: str = './dataset_cache/textvqa',
    ):
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
    raw_data = load_dataset('facebook/textvqa', trust_remote_code=True, cache_dir=cache_dir) #.select(range(10000))
    
    train_dataset = TextVQATrainDataset(
        raw_data['train'] 
    )
    test_dataset = TextVQATestDataset(
        raw_data['validation'],
    )
            
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=16, 
        shuffle=True,
        collate_fn=Qwen2DataCollator(processor)
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        num_workers=16, 
        collate_fn=Qwen2DataCollator(processor)
    )
    
    return train_dataloader, test_dataloader