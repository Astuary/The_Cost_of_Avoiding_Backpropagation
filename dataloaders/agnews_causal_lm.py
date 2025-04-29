import numpy as np
import torch 

from torch.utils.data import TensorDataset, Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer
)

class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels, class_labels=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = self._create_labels(labels)
        self.class_labels = class_labels
    
    def __len__(self):
        return len(self.labels)
    
    def _create_labels(self, labels):
        """Sets all labels to -100 except the last token per sequence."""
        last_token_indices = self.attention_mask.sum(dim=1) - 1  # Last valid token index

        for i in range(labels.shape[0]):
            labels[i, :last_token_indices[i]] = -100  # Mask everything except last
            labels[i, last_token_indices[i]+1:] = -100  # Mask everything except last

        return labels
    
    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        labels = self.labels[idx]
        if self.class_labels is not None:
            return input_ids, attention_mask, labels, self.class_labels[idx]
        return input_ids, attention_mask, labels

class TrainDataset(Dataset):
    def __init__(self, tokenizer, examples, loss_on_prefix=True):
        self.examples = examples
        self.qns = [ex["question"] for ex in self.examples]
        self.ans = [ex["answer"] for ex in self.examples]
        self.qns = tokenizer(self.qns, padding=False)
        self.ans = tokenizer(self.ans, padding=False)
        self.loss_on_prefix = loss_on_prefix
        self.max_len = max(
            [
                len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
                for i in range(len(self.examples))
            ]
        )
        # print(f"Max tokens: {self.max_len}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]
        pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
        
        tokens = qn_tokens + ans_tokens + pad_tokens
        mask = (
            ([int(self.loss_on_prefix)] * len(qn_tokens))
            + ([1] * len(ans_tokens))
            + ([0] * len(pad_tokens))
        )

        tokens = torch.tensor(tokens)
        mask = torch.tensor(mask)
        # ans_tokens = torch.tensor(ans_tokens)
        return tokens, mask#, ans_tokens

def get_centralized_dataset(
    train_batch_size: int = 8,
    test_batch_size: int = 80,
    max_seq_len: int = 400,
    random_seed: int = 0,
    cache_dir: str = './dataset_cache/agnews',
    tokenizer_name: str = 'bert-base-uncased',
    ):
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
    raw_datasets = load_dataset("SetFit/ag_news", cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # train_data = TrainDataset(
    #     tokenizer, raw_datasets['train']
    # )
    
    train_raw_text = np.array(raw_datasets['train']['text']) 
    test_raw_text = np.array(raw_datasets['test']['text'])
    
    train_raw_labels = np.array([
        label.replace('Sci/Tech', 'Science') for label in raw_datasets['train']['label_text']
    ])
    test_raw_labels = np.array([
        label.replace('Sci/Tech', 'Science') for label in raw_datasets['test']['label_text']
    ])
    test_class_labels = np.array([
        int(label) for label in raw_datasets['test']['label']
    ])
    
    tokenized_train_data = tokenizer(
        [
            "Text: " + i + "\nWhat class does this text fall under? (World, Science, Business, or Sports)" + "\n### Class: " + j.lower()
            for i, j in zip(train_raw_text, train_raw_labels)
        ], 
        max_length=max_seq_len,
        truncation=True,
        padding='max_length'
    )
    tokenized_test_data = tokenizer(
        [
            "Text: " + i + "\nWhat class does this text fall under? (World, Science, Business, or Sports)" + "\n### Class: "
            for i in test_raw_text
        ],  
        max_length=max_seq_len,
        truncation=True,
        padding='max_length'
    )
    
    tokenized_test_labels = tokenizer(
        [
            "Text: " + i + "\nWhat class does this text fall under? (World, Science, Business, or Sports)" + "\n### Class: " + j.lower()
            for i, j in zip(test_raw_text, test_raw_labels)
        ],  
        max_length=max_seq_len,
        truncation=True,
        padding='max_length'
    )
        
    train_input_ids = torch.tensor(tokenized_train_data['input_ids'], dtype=torch.int32)
    train_attention_mask = torch.tensor(tokenized_train_data['attention_mask'], dtype=torch.int32)
    train_labels = torch.tensor(tokenized_train_data['input_ids'])
    test_input_ids = torch.tensor(tokenized_test_data['input_ids'], dtype=torch.int32)
    test_attention_mask = torch.tensor(tokenized_test_data['attention_mask'], dtype=torch.int32)
    test_labels = torch.tensor(tokenized_test_labels['input_ids'])
    test_class_labels = torch.tensor(test_class_labels)
    
    train_data = CustomDataset(
        train_input_ids, train_attention_mask, train_labels
    )
    test_data = CustomDataset(
        test_input_ids, test_attention_mask, test_labels, test_class_labels
    )
    
    train_dataloader = DataLoader(
        train_data,
        batch_size=train_batch_size,
        num_workers=16, 
        shuffle=True
    )
        
    test_dataloader = DataLoader(
        test_data,
        batch_size=test_batch_size,
        num_workers=16
    )
    
    return train_dataloader, test_dataloader
