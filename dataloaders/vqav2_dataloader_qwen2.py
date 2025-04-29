import os
import sys

import json
import copy
from typing import Dict, Sequence

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence

from PIL import Image
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    PreTrainedTokenizer
)

class VQAv2TrainDataset(Dataset):
    def __init__(self, data, img_dir, processor, max_seq_len):
        self.img_dir = img_dir
        self.data = data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.processor = processor
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.data)
    
    def get_image_from_folder(self, name):
        image = Image.open(os.path.join(self.img_dir, 'COCO_train2014_' + str(name).zfill(12) + '.jpg'))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_tensor = self.transform(image)
        return image
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'image': self.get_image_from_folder(item['image_id']),
            'messages': item['messages'],
        }

class VQAv2TestDataset(Dataset):
    def __init__(self, data, img_dir, processor, max_seq_len):
        self.img_dir = img_dir
        self.data = data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.processor = processor
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.data)
    
    def get_image_from_folder(self, name):
        image = Image.open(os.path.join(self.img_dir, 'COCO_val2014_' + str(name).zfill(12) + '.jpg'))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_tensor = self.transform(image)
        return image
    
    def __getitem__(self, idx):
        item = self.data[idx]
        ground_truths = item['messages'][1]["content"][0]["text"]
        item['messages'][1]["content"][0]["text"] = ""
        return {
            'image': self.get_image_from_folder(item['image_id']),
            'messages': item['messages'],
            'answers': ground_truths
        }

class Qwen2DataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        
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
    test_batch_size: int = 6,
    max_seq_len: int = 100,
    random_seed: int = 0,
    cache_dir: str = './dataset_cache/vqav2',
    ):
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
    if not os.path.exists(cache_dir):
        print('Dataset not found at', cache_dir)
        sys.exit(0)
    else:
        train_data = []
        with open(cache_dir + '/finetunable_reformatted_huggingface_training_data_image_first.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                train_data.append(json.loads(line.strip()))
        test_data = []
        with open(cache_dir + '/finetunable_reformatted_huggingface_val_data_image_first.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line.strip()))
        
        train_dataset = VQAv2TrainDataset(train_data, cache_dir + '/train2014', processor, max_seq_len)
        test_dataset = VQAv2TestDataset(test_data, cache_dir + '/val2014', processor, max_seq_len)
        
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