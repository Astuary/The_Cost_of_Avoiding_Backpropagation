"""Load AG News data (training and eval)"""

import sys
import collections
from typing import Tuple

import torch
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer

from datasets import load_dataset

NUM_CLASSES = 4

def load_agnews_federated(
    dirichlet_parameter: float = 0.1,
    num_clients: int = 500,
    max_seq_len: int = 64,
    split: float = 0.8,
    random_seed: int = 0,
    cache_dir: str = './dataset_cache/agnews-seqcls',
    tokenizer_name: str = 'bert-base-uncased',
) -> Tuple[DataLoader, DataLoader]:
    """Construct a federated dataset from the centralized AG News.
    Sampling based on Dirichlet distribution over categories, following the paper
    Measuring the Effects of Non-Identical Data Distribution for
    Federated Visual Classification (https://arxiv.org/abs/1909.06335).
    Args:
      dirichlet_parameter: Parameter of Dirichlet distribution. Each client
        samples from this Dirichlet to get a multinomial distribution over
        classes. It controls the data heterogeneity of clients. If approaches 0,
        then each client only have data from a single category label. If
        approaches infinity, then the client distribution will approach IID
        partitioning.
      num_clients: The number of clients the examples are going to be partitioned
        on.
    Returns:
      A tuple of `torch.utils.data.DataLoader` representing unpreprocessed
      train data and test data.
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
    cache_dir = cache_dir + '-' + tokenizer_name
    raw_datasets = load_dataset("ag_news", cache_dir=cache_dir)
    raw_datasets = raw_datasets.shuffle(seed=42)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if 'Llama' in tokenizer_name:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = np.hstack(
        (np.array(raw_datasets['train']['text']), np.array(raw_datasets['test']['text'])),
        )
    labels = np.hstack(
        (np.array(raw_datasets['train']['label']), np.array(raw_datasets['test']['label'])),
        )
    
    example_count = len(inputs)
        
    train_clients = collections.OrderedDict()
    test_clients = collections.OrderedDict()

    multinomial_vals = []
    
    # Each client has a multinomial distribution over classes drawn from a
    # Dirichlet.
    for i in range(num_clients):
        proportion = np.random.dirichlet(
            dirichlet_parameter
            * np.ones(
                NUM_CLASSES,
            )
        )

        multinomial_vals.append(proportion)

    multinomial_vals = np.array(multinomial_vals)

    indices = []
    for k in range(NUM_CLASSES):
        label_k = np.where(labels == k)[0]
        np.random.shuffle(label_k)
        indices.append(label_k)

    example_indices = np.array(indices)

    client_samples = [[] for _ in range(num_clients)]
    count = np.zeros(NUM_CLASSES).astype(int)

    examples_per_client = (int(example_count / num_clients))
    examples_per_label = int(example_count / NUM_CLASSES)

    for k in range(num_clients):
        for i in range(examples_per_client):
            sampled_label = np.argwhere(
                np.random.multinomial(1, multinomial_vals[k, :]) == 1
            )[0][0]
            client_samples[k].append(
                example_indices[sampled_label, count[sampled_label]]
            )
            
            count[sampled_label] += 1
            
            if count[sampled_label] == examples_per_label:
                multinomial_vals[:, sampled_label] = 0
                multinomial_vals = (
                    multinomial_vals / multinomial_vals.sum(axis=1)[:, None]
                )

    for i in range(num_clients):
        client_name = str(i)
        x = inputs[np.array(client_samples[i])]
        y = (
            labels[np.array(client_samples[i])].astype("int64").squeeze()
        )
        
        tokenized_data = tokenizer(
            list(x), 
            max_length=max_seq_len,
            truncation=True,
            padding=True
        )
        
        all_input_ids = torch.tensor(tokenized_data['input_ids'])#, dtype=torch.int32)
        if 'token_type_ids' in tokenized_data.keys():
            all_token_type_ids = torch.tensor(tokenized_data['token_type_ids'])
        all_attention_mask = torch.tensor(tokenized_data['attention_mask'])#, dtype=torch.int32)
        all_labels = torch.tensor(y)
        
        split_count = int(all_input_ids.shape[0] * split)
        
        train_input_ids = all_input_ids[:split_count, :]
        if 'token_type_ids' in tokenized_data.keys():
            train_token_type_ids = all_token_type_ids[:split_count, :]
        train_attention_mask = all_attention_mask[:split_count, :]
        train_labels = all_labels[:split_count]
        
        test_input_ids = all_input_ids[split_count:, :]
        if 'token_type_ids' in tokenized_data.keys():
            test_token_type_ids = all_token_type_ids[split_count:, :]
        test_attention_mask = all_attention_mask[split_count:, :]
        test_labels = all_labels[split_count:]
        
        if 'token_type_ids' in tokenized_data.keys():
            train_data = TensorDataset(
                train_input_ids, train_token_type_ids, train_attention_mask, train_labels
            )
        else:
            train_data = TensorDataset(
                train_input_ids, train_attention_mask, train_labels
            )
            
        train_clients[client_name] = train_data
        
        if 'token_type_ids' in tokenized_data.keys():
            test_data = TensorDataset(
                test_input_ids, test_token_type_ids, test_attention_mask, test_labels
            )
        else:
            test_data = TensorDataset(
                test_input_ids, test_attention_mask, test_labels
            )
            
        test_clients[client_name] = test_data

    return train_clients, test_clients

def get_federated_datasets(
    dirichlet_parameter: float = 0.1,
    num_clients: int = 500,
    train_client_batch_size: int = 20,
    test_client_batch_size: int = 100,
    max_seq_len: int = 256,
    random_seed: int = 0,
    cache_dir: str = './dataset_cache/agnews-seqcls',
    tokenizer_name: str = 'bert-base-uncased',
):
    agnews_train_all_data, agnews_test_all_data = load_agnews_federated(
        dirichlet_parameter = dirichlet_parameter,
        num_clients = num_clients,
        cache_dir = cache_dir,
        max_seq_len = max_seq_len,
        tokenizer_name = tokenizer_name,
        random_seed = random_seed,
    )
    
    agnews_train_dataloader_dict = {}
    agnews_test_dataloader_dict = {}
    
    for client in range(num_clients):
        
        agnews_train_dataloader_dict[str(client)] = DataLoader(
            agnews_train_all_data[str(client)],
            batch_size=train_client_batch_size,
            num_workers=2
        )
        
        agnews_test_dataloader_dict[str(client)] = DataLoader(
            agnews_test_all_data[str(client)],
            batch_size=test_client_batch_size,
            num_workers=2
        )
    
    return agnews_train_dataloader_dict, agnews_test_dataloader_dict
    