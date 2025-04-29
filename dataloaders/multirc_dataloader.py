import numpy as np
import torch 

from torch.utils.data import TensorDataset, Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer
)

class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        labels = self.labels[idx]
        return input_ids, attention_mask, labels

def get_few_shot_example(question, choices):
    few_shots = "This is an example on how to solve the following multi-choice problem:\
    Davis decided to kill Adams. He set out for Adams's house. Before he got there he saw Brooks, who resembled Adams. Thinking that Brooks was Adams, Davis shot at Brooks. The shot missed Brooks but wounded Case, who was some distance away. Davis had not seen Case. In a prosecution under a statute that proscribes any attempt to commit murder, the district attorney should indicate that the intended victim(s) was/were\
    . Pick one of these choices (index starts at 0): [Adams only., Brooks only., Case only., Adams and Brooks].\
    . The right answer is choice #### 1.\n"
    few_shots += "Now to answer this question: "
    few_shots += question
    few_shots += ". Pick one of these choices (index starts at 0): "
    few_shots += str(choices)
    few_shots += ". The right answer is choice #### "
    
    return few_shots

def get_centralized_dataset(
    train_batch_size: int = 4,
    test_batch_size: int = 40,
    max_seq_len: int = 1500,
    random_seed: int = 0,
    cache_dir: str = './dataset_cache/multirc',
    tokenizer_name: str = 'bert-base-uncased',
    ):
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
    raw_datasets = load_dataset('super_glue', 'multirc', trust_remote_code=True, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
        
    def filter_invalid_labels(example):
        return example["label"] != -1

    train_data = raw_datasets['train']#.filter(filter_invalid_labels)
    test_data = raw_datasets['validation']#.filter(filter_invalid_labels)
    
    train_raw_paragraph = np.array(train_data['paragraph']) 
    test_raw_paragraph = np.array(test_data['paragraph'])
    
    train_raw_question = np.array(train_data['question']) 
    test_raw_question = np.array(test_data['question'])
    
    train_raw_answer = np.array(train_data['answer'])
    test_raw_answer = np.array(test_data['answer'])
    
    train_raw_label = np.array(train_data['label'])
    test_raw_label = np.array(test_data['label'])
    
    tokenized_train_data = tokenizer(
        [
            "### Instruction: Given a paragraph (P), check whether the answer (A) to the given question (Q) is correct." +
            "\nP: " + p + "\nQ: " + q + "\nA: " + a
            for p, q, a in zip(train_raw_paragraph, train_raw_question, train_raw_answer)
        ], 
        max_length=max_seq_len,
        truncation=True,
        padding='max_length'
    )
    tokenized_test_data = tokenizer(
        [
            "### Instruction: Given a paragraph (P), check whether the answer (A) to the given question (Q) is correct." +
            "\nP: " + p + "\nQ: " + q + "\nA: " + a
            for p, q, a in zip(test_raw_paragraph, test_raw_question, test_raw_answer)
        ], 
        max_length=max_seq_len,
        truncation=True,
        padding='max_length'
    )
    
    train_input_ids = torch.tensor(tokenized_train_data['input_ids'],)# dtype=torch.int64)
    train_attention_mask = torch.tensor(tokenized_train_data['attention_mask'],)# dtype=torch.int32)
    train_labels = torch.tensor(train_raw_label)
    print('Training sample count:', len(train_labels))
    test_input_ids = torch.tensor(tokenized_test_data['input_ids'],)# dtype=torch.int64)
    test_attention_mask = torch.tensor(tokenized_test_data['attention_mask'],)# dtype=torch.int32)
    test_labels = torch.tensor(test_raw_label)
    print('Testing sample count:', len(test_labels))
    
    train_data = CustomDataset(
        train_input_ids, train_attention_mask, train_labels
    )
    test_data = CustomDataset(
        test_input_ids, test_attention_mask, test_labels
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
