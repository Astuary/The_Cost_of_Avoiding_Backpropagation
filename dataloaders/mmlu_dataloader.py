import numpy as np
import torch 

from torch.utils.data import TensorDataset, Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer
)

class TestDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels, answers):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.answers = answers
    
    def __len__(self):
        return len(self.answers)
    
    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        labels = self.labels[idx]
        answers = self.answers[idx]
        return input_ids, attention_mask, labels, answers

class TrainDataset(Dataset):
    def __init__(self, tokenizer, examples, loss_on_prefix=True):
        self.examples = examples
        self.qns = ["To answer this question: " + ex["question"] for ex in self.examples]
        self.chc = [". Pick one of these choices (index starts at 0): " + str(ex["choices"]) for ex in self.examples]
        self.ans = [". The right answer is choice #### " + str(ex["answer"]) for ex in self.examples]
        
        self.qns = tokenizer(self.qns, padding=False)
        self.chc = tokenizer(self.chc, padding=False)
        self.ans = tokenizer(self.ans, padding=False)
        
        self.loss_on_prefix = loss_on_prefix
        self.max_len = max(
            [
                len(self.qns["input_ids"][i]) + len(self.chc["input_ids"][i]) + len(self.ans["input_ids"][i])
                for i in range(len(self.examples))
            ]
        )
        print(f"Max tokens: {self.max_len}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        chc_tokens = self.chc["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]
        pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(chc_tokens) - len(ans_tokens))
        
        tokens = qn_tokens + chc_tokens + ans_tokens + pad_tokens
        mask = (
            ([int(self.loss_on_prefix)] * len(qn_tokens))
            + ([1] * len(chc_tokens))
            + ([1] * len(ans_tokens))
            + ([0] * len(pad_tokens))
        )

        tokens = torch.tensor(tokens)
        mask = torch.tensor(mask)
        return tokens, mask

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
    test_batch_size: int = 20,
    max_seq_len: int = 1500,
    random_seed: int = 0,
    cache_dir: str = './dataset_cache/mmlu',
    tokenizer_name: str = 'bert-base-uncased',
    ):
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
    raw_datasets = load_dataset("cais/mmlu", "all", cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # train_data = TrainDataset(
    #     tokenizer, raw_datasets['auxiliary_train']
    # )
    
    train_raw_inputs = np.array(raw_datasets['auxiliary_train']['question']) 
    test_raw_inputs = np.array(raw_datasets['test']['question'])
    
    train_raw_choices = np.array(raw_datasets['auxiliary_train']['choices']) 
    test_raw_choices = np.array(raw_datasets['test']['choices'])
    
    train_raw_labels = np.array(raw_datasets['auxiliary_train']['answer'])
    test_raw_labels = np.array(raw_datasets['test']['answer'])
    
    subset_train_samples = []
    subset_train_labels = []
    for q, c, a in zip(train_raw_inputs, train_raw_choices, train_raw_labels):
        if len(q) < 1000:
            subset_train_samples.append(
                "To answer this question: " + q
                + " Pick one of these choices from [A, B, C, D]. "
                + "A: " + list(c)[0] + " "
                + "B: " + list(c)[1] + " "
                + "C: " + list(c)[2] + " "
                + "D: " + list(c)[3] + " "
                + ". The right answer is choice #### "
                + {0: "A", 1: "B", 2: "C", 3: "D"}[int(a)]
            )
            subset_train_labels.append(
                {0: "A", 1: "B", 2: "C", 3: "D"}[int(a)]
            )
    
    subset_test_samples = []
    subset_test_labels = []
    for q, c, a in zip(test_raw_inputs, test_raw_choices, test_raw_labels):
        if len(q) < 500:
            subset_test_samples.append(
                "To answer this question: " + q
                + " Pick one of these choices from [A, B, C, D]. "
                + "A: " + list(c)[0] + " "
                + "B: " + list(c)[1] + " "
                + "C: " + list(c)[2] + " "
                + "D: " + list(c)[3] + " "
                + ". The right answer is choice #### "
            )
            subset_test_labels.append(
                {0: "A", 1: "B", 2: "C", 3: "D"}[int(a)]
            )
    
    print('Number of qualified train samples: ', len(subset_train_samples))
    print('Number of qualified test samples: ', len(subset_test_samples))

    tokenized_train_data = tokenizer(
        subset_train_samples, 
        max_length=max_seq_len,
        truncation=True,
        padding='max_length'
    )
    tokenized_test_data = tokenizer(
        subset_test_samples, 
        max_length=1000,
        truncation=True,
        padding='max_length'
    )
    
    tokenized_train_labels = tokenizer(
        subset_train_labels,
        max_length=2,
        truncation=True,
        padding='max_length'
    )
    tokenized_test_labels = tokenizer(
        subset_test_labels,
        max_length=2,
        truncation=True,
        padding='max_length'
    )
    
    train_input_ids = torch.tensor(tokenized_train_data['input_ids'], dtype=torch.int64)
    train_attention_mask = torch.tensor(tokenized_train_data['attention_mask'], dtype=torch.int32)
    train_labels = torch.tensor(tokenized_train_labels['input_ids'])
    test_input_ids = torch.tensor(tokenized_test_data['input_ids'], dtype=torch.int64)
    test_attention_mask = torch.tensor(tokenized_test_data['attention_mask'], dtype=torch.int32)
    test_labels = torch.tensor(tokenized_test_labels['input_ids'])
    
    train_data = TestDataset(
        train_input_ids, train_attention_mask, train_labels, subset_train_labels
    )
    test_data = TestDataset(
        test_input_ids, test_attention_mask, test_labels, subset_test_labels
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
