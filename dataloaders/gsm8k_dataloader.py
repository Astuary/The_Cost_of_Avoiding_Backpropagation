import numpy as np
import torch 

from torch.utils.data import TensorDataset, Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer
)

class CustomDataset(Dataset):
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
        return tokens, mask#, ans_tokens

def get_few_shot_example(question):
    few_shots = ""
    # #1
    # few_shots += "Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\
    # A: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72.\n"
    # #2
    # few_shots += "Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\n\
    # A: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10.\n"
    # #3
    # few_shots += "Q: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?\n\
    # A: He writes each friend 3*2=<<3*2=6>>6 pages a week So he writes 6*2=<<6*2=12>>12 pages every week That means he writes 12*52=<<12*52=624>>624 pages a year #### 624.\n"
    # #4
    # few_shots += "Q: Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?\n\
    # A: Half of the number of Randy's mango trees is 60/2 = <<60/2=30>>30 trees. So Randy has 30 - 5 = <<30-5=25>>25 coconut trees. Therefore, Randy has 60 + 25 = <<60+25=85>>85 treeson his farm. #### 85.\n"
    # #5
    # few_shots += "Q: John writes 20 pages a day. How long will it take him to write 3 books that are 400 pages each?\n\
    # A: He wants to write 3*400=<<3*400=1200>>1200 pages So it will take him 1200/20=<<1200/20=60>>60 days #### 60\n"
    # #6
    # few_shots += "Q: Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?\n\
    # A: In one hour, there are 3 sets of 20 minutes. So, Joy can read 8 x 3 = <<8*3=24>>24 pages in an hour. It will take her 120/24 = <<120/24=5>>5 hours to read 120 pages. #### 5\n"
    # #7
    # few_shots += "Q: There are 25 roses in a garden. There are 40 tulips. There are 35 daisies. What percentage of flowers are not roses?\n\
    # A: There are 25+40+35=<<25+40+35=100>>100 flowers total. There are 40+35=<<40+35=75>>75 flowers that are not roses. Therefore, (75/100)*100=<<(75/100)*100=75>>75% of the flowers are not roses. #### 75\n"
    # #8
    # few_shots += "Q: Tim has 30 less apples than Martha, and Harry has half as many apples as Tim. If Martha has 68 apples, how many apples does Harry have?\n\
    # A: Tim has 68-30 = <<68-30=38>>38 apples. Harry has 38/2 = <<38/2=19>>19 apples. #### 19\n"
    # few_shots += "### Instruction: \nSolve the following math problem, given the previous examples.\n"
    # few_shots += "### Instruction: \nSolve the following math problem by thinking step by step, given the previous examples. Give the final answer in form of a number:\n"
    few_shots += "Q: "
    few_shots += question
    few_shots += "\n### A: "
        
    return few_shots

def get_centralized_dataset(
    train_batch_size: int = 8,
    test_batch_size: int = 40,
    max_seq_len: int = 800,
    random_seed: int = 0,
    cache_dir: str = './dataset_cache/gsm8k',
    tokenizer_name: str = 'bert-base-uncased',
    ):
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
    raw_datasets = load_dataset("openai/gsm8k", "main", cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # train_data = TrainDataset(
    #     tokenizer, raw_datasets['train']
    # )
    
    train_raw_inputs = np.array(raw_datasets['train']['question']) 
    test_raw_inputs = np.array(raw_datasets['test']['question'])
    
    train_raw_labels = np.array(raw_datasets['train']['answer'])
    test_raw_labels = np.array(raw_datasets['test']['answer'])
    
    tokenized_train_data = tokenizer(
        [
            # "### Instruction: \nSolve the following math problem by thinking step by step:\n### Input: \n" + i + "\n### The answer is: \n" + j
            # "### Instruction: \nSolve the following math problem by thinking step by step:\n Q: \n" + i + "\n### A: \n" + j
            "Q: " + i + "\n### A: " + j
            for i, j in zip(train_raw_inputs, train_raw_labels)
        ], 
        max_length=max_seq_len,
        truncation=True,
        padding='max_length'
    )
    tokenized_test_data = tokenizer(
        # list(test_raw_inputs), 
        [get_few_shot_example(i) for i in test_raw_inputs], 
        max_length=600,
        truncation=True,
        padding='max_length'
    )
    
    tokenized_train_labels = tokenizer(
        list(train_raw_labels),
        # [l.split('####')[1] for l in train_raw_labels], 
        max_length=max_seq_len,
        truncation=True,
        padding='max_length'
    )
    tokenized_test_labels = tokenizer(
        list(test_raw_labels),
        # [l.split('####')[1] for l in test_raw_labels], 
        max_length=max_seq_len,
        truncation=True,
        padding='max_length'
    )
    
    train_input_ids = torch.tensor(tokenized_train_data['input_ids'])#, dtype=torch.int64)
    train_attention_mask = torch.tensor(tokenized_train_data['attention_mask'])#, dtype=torch.int32)
    train_labels = torch.tensor(tokenized_train_labels['input_ids'])
    test_input_ids = torch.tensor(tokenized_test_data['input_ids'])#, dtype=torch.int32)
    test_attention_mask = torch.tensor(tokenized_test_data['attention_mask'])#, dtype=torch.int32)
    test_labels = torch.tensor(tokenized_test_labels['input_ids'])
    
    train_data = CustomDataset(
        train_input_ids, train_attention_mask, train_labels, train_raw_labels
    )
    test_data = CustomDataset(
        test_input_ids, test_attention_mask, test_labels, [l.split('####')[1] for l in test_raw_labels]
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
