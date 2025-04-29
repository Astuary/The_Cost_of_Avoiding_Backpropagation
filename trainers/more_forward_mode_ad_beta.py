"""Centralized implementation of Forward-mode Automatic Differentiation"""
import os
import sys

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)
# print(sys.path)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = 'false'
os.environ["TRANSFORMERS_VERBOSITY"] = 'error'
# os.environ["WANDB_MODE"] = "disabled"

import math
import copy
import json
import time
import wandb
import socket
import random
import argparse
from tqdm import tqdm
from datetime import datetime, timedelta
from functools import partial

import torch
import numpy as np
import functorch as ft
import torch.func as fc
from torch.optim import AdamW, Adam, SGD, NAdam
from torch.autograd.profiler import record_function
import torch.autograd.forward_ad as fwAD

import dataloaders.agnews_sequence_classification_dataloader as agnews_sequence_classification_dataloader
import dataloaders.gsm8k_dataloader as gsm8k_dataloader
import dataloaders.mmlu_dataloader as mmlu_dataloader
import dataloaders.vqav2_dataloader_qwen2 as vqav2_dataloader
import dataloaders.boolq_dataloader as boolq_dataloader
import dataloaders.multirc_dataloader as multirc_dataloader
import dataloaders.gqa_dataloader as gqa_dataloader
import dataloaders.textvqa_dataloader as textvqa_dataloader

from evaluate import load as load_metric
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    LlavaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    GPTQConfig,
    AwqConfig
)

from utils.evaluation import *
from utils.function_call_tracker import *
from utils.matrix_operations import *
from utils.forward_mode_samplers import *
from utils.perturbation_calibration import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
if DEVICE == 'cuda':
    GPU_NAME = torch.cuda.get_device_name(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def trace_handler(prof: torch.profiler.profile):
   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   
   if not os.path.exists(f"{results_dir}memory_profile"):
        os.makedirs(f"{results_dir}memory_profile")
        
   file_prefix = f"{results_dir}memory_profile/{host_name}_{timestamp}"

   # Construct the trace file.
   prof.export_chrome_trace(f"{file_prefix}.json.gz")

   # Construct the memory timeline file.
   prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")

def calibrated_gradients_text_generation_train(net, train_dataloader, test_dataloader):
    if optimizer_name =='adamw': 
        optimizer = AdamW(net.parameters(), lr=lr)
    elif optimizer_name =='adam': 
        optimizer = Adam(net.parameters(), lr=lr)
    elif optimizer_name =='sgd': 
        optimizer = SGD(net.parameters(), lr=lr)
    elif optimizer_name =='sgd_nesterov': 
        optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=True)
    
    losses_epoch = {}
    losses_iter = {}
    accuracies = {}
    times_loss = {}
    times_accuracy = {}
    iteration = 0
    
    # print('Before the training.', flush=True)
    # print('Memory allocated: ', torch.cuda.max_memory_allocated() / (1024 ** 3), flush=True)
    # print('Memory reserved: ', torch.cuda.max_memory_reserved() / (1024 ** 3), flush=True)
    
    params = {name: p for name, p in net.named_parameters() if p.requires_grad}
    trainable_layer_names = list(params.keys())
    
    for epoch in range(1, epochs+1):
        print(f'Epoch {epoch}, Iteration {iteration}')
        net.train()
        # with torch.profiler.profile(
        #     activities=[
        #         torch.profiler.ProfilerActivity.CPU,
        #         torch.profiler.ProfilerActivity.CUDA,
        #     ],
        #     schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True,
        #     on_trace_ready=trace_handler,
        # ) as prof:
        accumulated_loss = 0.0
        start_time = time.time()
        sample_count = 0
        for batch_index, batch in enumerate(train_dataloader):
            # prof.step()
            if len(batch) == 3:
                batch_dict = {
                    'input_ids': batch[0].to(DEVICE),
                    'attention_mask': batch[1].to(DEVICE),
                    'labels': batch[0].to(DEVICE),
                }
            else:
                batch_dict = {
                    'input_ids': batch[0].to(DEVICE),
                    'token_type_ids': batch[1].to(DEVICE),
                    'attention_mask': batch[2].to(DEVICE),
                    'labels': batch[0].to(DEVICE),
                }
                
            # torch.cuda.reset_max_memory_allocated()
            # start_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            # start_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            
            for perturbation_index in range(perturbation_count): 
                perturbation_seed = random.randint(0, 2**31 - 1)
                if fixed_seed:
                    torch.manual_seed(perturbation_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(perturbation_seed)
                        torch.cuda.manual_seed_all(perturbation_seed)
    
                if iteration == 0:
                    gradients, tangents, loss, jvp = generate_gradients_for_calibration_phase(
                        net, trainable_layer_names, 
                        params, batch_dict, perturbation_var, jvp_range
                    )
                    
                    accumulated_loss += loss.item()
                    for n, p in params.items():
                        if p.grad == None:
                            p.grad = (gradients[n]).to(p.dtype)
                        else:
                            p.grad += (gradients[n]).to(p.dtype)
                        if torch.isnan(p.grad).any():
                            print('Iteration #: ', iteration, 'NaN Gradient', flush=True)
                        if torch.isnan(p).any():
                            print('Iteration #: ', iteration, 'NaN Weight', flush=True)
                    
                else:
                    tangents = {
                        name: (torch.randn_like(p) * torch.sqrt(torch.tensor(perturbation_var))).to(DEVICE) 
                        for name, p in params.items()
                    }
                
                    with fwAD.dual_level(), torch.no_grad(), torch.cuda.amp.autocast():
                        new_state_dict = {}
                        for n, v in tangents.items():
                            new_state_dict[n] = fwAD.make_dual(params[n], v)
                        net.load_state_dict(new_state_dict, strict=False)
                        del new_state_dict
                        
                        outputs = net(**batch_dict)
                        loss = outputs.loss
                        accumulated_loss += loss.item()
                        
                        jvp = fwAD.unpack_dual(loss).tangent
                        if jvp_range > 0.0:
                            jvp = torch.clamp(jvp, min=-jvp_range, max=jvp_range)
                        # print('jvp: ', jvp, flush=True)
                    
                    if torch.isnan(jvp):
                        print('Iteration #:', iteration, 'Skipping NaN jvp', flush=True)
                        continue
                    
                    for p, v in zip(params.values(), tangents.values()):
                        if p.grad == None:
                            p.grad = (v * jvp).to(p.dtype)
                        else:
                            p.grad += (v * jvp).to(p.dtype)
                        if torch.isnan(p.grad).any():
                            print('Iteration #: ', iteration, 'NaN Gradient', flush=True)
                        if torch.isnan(p).any():
                            print('Iteration #: ', iteration, 'NaN Weight', flush=True)
                
            for p in params.values():
                if p.grad is not None:
                    p.grad /= perturbation_count
            
            wandb.log({
                "loss": loss.item(),
                "jvp": jvp.item(),
                "gradients_mean": {
                    name: param.grad.mean().item() for name, param in net.named_parameters() if param.grad is not None 
                },
                "gradients_std": {
                    name: param.grad.std().item() for name, param in net.named_parameters() if param.grad is not None 
                },
                "gradients_norm": {
                    name: torch.norm(param.grad).item() for name, param in net.named_parameters() if param.grad is not None
                },
                "gradients_max": {
                    name: param.grad.max().item() for name, param in net.named_parameters() if param.grad is not None
                },
                "gradients_min": {
                    name: param.grad.min().item() for name, param in net.named_parameters() if param.grad is not None
                },
                "random_gradient": dict(net.named_parameters())[key_to_track].grad[0][0] if dict(net.named_parameters())[key_to_track].grad is not None else 0,
                "perturbations_mean": {
                    name: pert.mean().item() for name, pert in tangents.items()
                },
                "perturbations_std": {
                    name: pert.std().item() for name, pert in tangents.items()
                },
                "perturbations_max": {
                    name: pert.max().item() for name, pert in tangents.items()
                },
                "perturbations_min": {
                    name: pert.min().item() for name, pert in tangents.items()
                },
                "perturbations_norm": {
                    name: torch.norm(pert).item() for name, pert in tangents.items()
                },
                "random_perturbation": dict(tangents.items())[key_to_track][0][0],
            })
            del tangents
            
            # end_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            # end_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            
            # print('Memory allocated: ', end_memory_allocated - start_memory_allocated, flush=True)    
            # print('Memory reserved: ', end_memory_reserved - start_memory_reserved, flush=True)    
            with record_function("## optimizer ##"):
                optimizer.step()
                optimizer.zero_grad()
                                            
            sample_count += batch_dict['input_ids'].shape[0]
            iteration += 1
            
            losses_iter[iteration] = loss.item() / batch_dict['input_ids'].shape[0] 
            
            wandb.log({
                "parameters_mean": {
                    name: param.data.mean().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_std": {
                    name: param.data.std().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_norm": {
                    name: torch.norm(param.data).item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_max": {
                    name: param.data.max().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_min": {
                    name: param.data.min().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "random_parameter": dict(net.named_parameters())[key_to_track][0][0],
            })
            
            if batch_index >= max_batches: 
                break
                    
        end_time = time.time() 
        print(f'Epoch {epoch} training time: {end_time - start_time}')
        
        start_time = time.time() 
        if dataset == 'gsm8k':
            _, accuracy, _ = text_generation_gsm8k_test(net, test_dataloader)
        elif dataset == 'mmlu':
            _, accuracy, _ = text_generation_mmlu_test(net, test_dataloader)
        end_time = time.time() 
        print(f'Epoch {epoch} test time: {end_time - start_time}')
        
        losses_epoch[epoch] = accumulated_loss / sample_count
        accuracies[epoch] = accuracy
        times_accuracy[end_time - start_time] = accuracy
        times_loss[end_time - start_time] = accumulated_loss / sample_count
        
        print(f'Epoch {epoch} accuracy: {accuracy}')
    
        if epoch % checkpoint_interval == 0:
            with open(results_dir + 'train_losses_epoch.json', 'w+') as f:
                json.dump(losses_epoch, f)
            with open(results_dir + 'train_losses_iter.json', 'w+') as f:
                json.dump(losses_iter, f)
            with open(results_dir + 'accuracies.json', 'w+') as f:
                json.dump(accuracies, f)
            with open(results_dir + 'times_accuracy.json', 'w+') as f:
                json.dump(times_accuracy, f)
            with open(results_dir + 'times_loss.json', 'w+') as f:
                json.dump(times_loss, f)
            
        print('')

def gradient_accumulation_text_generation_train(net, train_dataloader, test_dataloader):
    if optimizer_name =='adamw': 
        optimizer = AdamW(net.parameters(), lr=lr)
    elif optimizer_name =='adam': 
        optimizer = Adam(net.parameters(), lr=lr)
    elif optimizer_name =='nadam': 
        optimizer = NAdam(net.parameters(), lr=lr, momentum_decay=momentum)
    elif optimizer_name =='sgd': 
        optimizer = SGD(net.parameters(), lr=lr)
    elif optimizer_name =='sgd_nesterov': 
        optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=True)
    
    losses_epoch = {}
    losses_iter = {}
    accuracies = {}
    times_loss = {}
    times_accuracy = {}
    iteration = 0
    valid_iteration = 0
    
    # print('Before the training.', flush=True)
    # print('Memory allocated: ', torch.cuda.max_memory_allocated() / (1024 ** 3), flush=True)
    # print('Memory reserved: ', torch.cuda.max_memory_reserved() / (1024 ** 3), flush=True)
    
    params = {name: p for name, p in net.named_parameters() if p.requires_grad}
    
    for epoch in range(1, epochs+1):
        print(f'Epoch {epoch}, Iteration {iteration}')
        net.train()
        # with torch.profiler.profile(
        #     activities=[
        #         torch.profiler.ProfilerActivity.CPU,
        #         torch.profiler.ProfilerActivity.CUDA,
        #     ],
        #     schedule=torch.profiler.schedule(wait=0, warmup=0, active=25, repeat=1),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True,
        #     on_trace_ready=trace_handler,
        # ) as prof:
        accumulated_loss = 0.0
        start_time = time.time()
        sample_count = 0
        for batch_index, batch in enumerate(train_dataloader):
            # prof.step()
            batch_dict = {
                'input_ids': batch[0].to(DEVICE),
                'attention_mask': batch[1].to(DEVICE),
                'labels': batch[0].to(DEVICE),
                'use_cache': False,
            }
            
            # torch.cuda.reset_max_memory_allocated()
            # start_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            # start_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            
            if fixed_seed:
                perturbation_seed = random.randint(0, 2**31 - 1)
                torch.manual_seed(perturbation_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(perturbation_seed)
                    torch.cuda.manual_seed_all(perturbation_seed)

            tangents = {
                name: (torch.randn_like(p) * torch.sqrt(torch.tensor(perturbation_var))).to(DEVICE) 
                for name, p in params.items()
            }
            
            with fwAD.dual_level(), torch.no_grad(), record_function("## forward ##"):
                new_state_dict = {}
                for n, p in params.items():
                    new_state_dict[n] = fwAD.make_dual(p, tangents[n])
                net.load_state_dict(new_state_dict, strict=False)
                del new_state_dict
                
                outputs = net(**batch_dict)
                loss = outputs.loss
                accumulated_loss += loss.item()
                
                jvp = fwAD.unpack_dual(loss).tangent
                if jvp_range > 0.0:
                    jvp = torch.clamp(jvp, min=-jvp_range, max=jvp_range)
                
                if torch.isnan(jvp):
                    print('Iteration #:', iteration, 'Skipping NaN jvp', flush=True)
                    continue
                else:
                    valid_iteration += 1
                
            for p, v in zip(params.values(), tangents.values()):
                if p.grad == None:
                    p.grad = v * jvp
                else:
                    p.grad += v * jvp
                if torch.isnan(p.grad).any():
                    print('Iteration #: ', iteration, 'NaN Gradient', flush=True)
                if torch.isnan(p).any():
                    print('Iteration #: ', iteration, 'NaN Weight', flush=True)
                
            if valid_iteration % accumulation_steps == 0:       
                for p in params.values():
                    p.grad /= accumulation_steps
                        
                if grad_norm_range > 0.0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_norm_range)
                if grad_value_range > 0.0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_value_range)
            
                wandb.log({
                    "loss": loss.item(),
                    "jvp": jvp.item(),
                    "gradients_mean": {
                        name: param.grad.mean().item() for name, param in net.named_parameters() if param.grad is not None 
                    },
                    "gradients_std": {
                        name: param.grad.std().item() for name, param in net.named_parameters() if param.grad is not None 
                    },
                    "gradients_norm": {
                        name: torch.norm(param.grad).item() for name, param in net.named_parameters() if param.grad is not None
                    },
                    "gradients_max": {
                        name: param.grad.max().item() for name, param in net.named_parameters() if param.grad is not None
                    },
                    "gradients_min": {
                        name: param.grad.min().item() for name, param in net.named_parameters() if param.grad is not None
                    },
                    "random_gradient": dict(net.named_parameters())[key_to_track].grad[0][0] if dict(net.named_parameters())[key_to_track].grad is not None else 0,
                    "perturbations_mean": {
                        name: pert.mean().item() for name, pert in tangents.items()
                    },
                    "perturbations_std": {
                        name: pert.std().item() for name, pert in tangents.items()
                    },
                    "perturbations_max": {
                        name: pert.max().item() for name, pert in tangents.items()
                    },
                    "perturbations_min": {
                        name: pert.min().item() for name, pert in tangents.items()
                    },
                    "perturbations_norm": {
                        name: torch.norm(pert).item() for name, pert in tangents.items()
                    },
                    "random_perturbation": dict(tangents.items())[key_to_track][0][0],
                })
                del tangents
            
                # end_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
                # end_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
                
                # print('Memory allocated: ', end_memory_allocated - start_memory_allocated, flush=True)    
                # print('Memory reserved: ', end_memory_reserved - start_memory_reserved, flush=True)    
                with record_function("## optimizer ##"):
                    optimizer.step()
                    optimizer.zero_grad()
                
            sample_count += batch_dict['input_ids'].shape[0]
            iteration += 1
            
            losses_iter[iteration] = loss.item() / batch_dict['input_ids'].shape[0] 
            
            wandb.log({
                "parameters_mean": {
                    name: param.data.mean().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_std": {
                    name: param.data.std().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_norm": {
                    name: torch.norm(param.data).item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_max": {
                    name: param.data.max().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_min": {
                    name: param.data.min().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "random_parameter": dict(net.named_parameters())[key_to_track][0][0],
            })
            
            if batch_index >= max_batches: 
                break
                    
        end_time = time.time() 
        print(f'Epoch {epoch} training time: {end_time - start_time}')
           
        start_test_time = time.time() 
        if dataset == 'gsm8k':
            _, accuracy, _ = text_generation_gsm8k_test(net, test_dataloader)
        elif dataset == 'mmlu':
            _, accuracy, _ = text_generation_mmlu_test(net, test_dataloader)
        end_test_time = time.time() 
        print(f'Epoch {epoch} test time: {end_test_time - start_test_time}')
        
        losses_epoch[epoch] = accumulated_loss / sample_count
        accuracies[epoch] = accuracy
        times_accuracy[end_time - start_time] = accuracy
        times_loss[end_time - start_time] = accumulated_loss / sample_count
        
        print(f'Epoch {epoch} accuracy: {accuracy}')
    
        if epoch % checkpoint_interval == 0:
            with open(results_dir + 'train_losses_epoch.json', 'w+') as f:
                json.dump(losses_epoch, f)
            with open(results_dir + 'train_losses_iter.json', 'w+') as f:
                json.dump(losses_iter, f)
            with open(results_dir + 'accuracies.json', 'w+') as f:
                json.dump(accuracies, f)
            with open(results_dir + 'times_accuracy.json', 'w+') as f:
                json.dump(times_accuracy, f)
            with open(results_dir + 'times_loss.json', 'w+') as f:
                json.dump(times_loss, f)
            
        print('')

def text_generation_gsm8k_test(net, test_dataloader):
    correct = 0
    accumulated_loss = 0
    sample_count = 0
    net.eval()
    
    with torch.no_grad():
        for batch in test_dataloader:
            batch_dict = {
                'input_ids': batch[0].to(DEVICE),
                'attention_mask': batch[1].to(DEVICE),
                # 'labels': batch[2].to(DEVICE),
            }
            
            try:
                prediction_tokens = net.generate(
                    **batch_dict, 
                    max_new_tokens=150,
                    # num_beams=4,
                    # do_sample=True,
                    temperature=0.7,
                    top_p = 0.95,
                    # top_k = 1,
                    # repetition_penalty=1,
                )
                predictions = tokenizer.batch_decode(prediction_tokens, skip_special_tokens=True)
                extracted_answers = gsm8k_extract_pattern_from_array(predictions)
                
                for extracted_answer, ground_truth in zip(extracted_answers, batch[3]):
                    if ground_truth.strip() in extracted_answer:#.split(' '):
                        correct += 1
            except Exception as e:
                print(e)
            
            sample_count += batch[0].shape[0]
        
    accumulated_loss /= sample_count
    accuracy = correct / sample_count # metric.compute()["accuracy"]
    print('Test accuracy: ', accuracy)
    
    return accumulated_loss, accuracy, sample_count

def text_generation_mmlu_test(net, test_dataloader):
    correct = 0
    accumulated_loss = 0
    sample_count = 0
    net.eval()
    
    abcd_indices = [
        tokenizer('A', add_special_tokens=False).input_ids[0],
        tokenizer('a', add_special_tokens=False).input_ids[0],
        tokenizer('B', add_special_tokens=False).input_ids[0],
        tokenizer('b', add_special_tokens=False).input_ids[0],
        tokenizer('C', add_special_tokens=False).input_ids[0],
        tokenizer('c', add_special_tokens=False).input_ids[0],
        tokenizer('D', add_special_tokens=False).input_ids[0],
        tokenizer('d', add_special_tokens=False).input_ids[0],
    ]
    
    with torch.no_grad():
        for batch_index, batch in enumerate(test_dataloader):
            batch_dict = {
                'input_ids': batch[0].to(DEVICE),
                'attention_mask': batch[1].to(DEVICE),
                # 'labels': batch[2].to(DEVICE),
            }
            
            prediction_tokens = net.generate(
                **batch_dict, 
                max_new_tokens=50,
                output_scores=True,
                return_dict_in_generate=True,
                # num_beams=4,
                # do_sample=True,
                temperature=0.7,
                top_p = 1, #0,
                # top_k = 1,
                # repetition_penalty=1,
            )
            logits = prediction_tokens.scores
            predictions = tokenizer.batch_decode(prediction_tokens.sequences, skip_special_tokens=True)
                
            extracted_answers = mmlu_extract_letters_after_hashes(predictions)
            
            for i, (extracted_answer, full_answer) in enumerate(zip(extracted_answers, predictions)):
                a_prob = logits[0][i][abcd_indices[0]] + logits[0][i][abcd_indices[1]]
                b_prob = logits[0][i][abcd_indices[2]] + logits[0][i][abcd_indices[3]]
                c_prob = logits[0][i][abcd_indices[4]] + logits[0][i][abcd_indices[5]]
                d_prob = logits[0][i][abcd_indices[6]] + logits[0][i][abcd_indices[7]]
                argmax_prob = np.argmax([a_prob.cpu(), b_prob.cpu(), c_prob.cpu(), d_prob.cpu()]) 
                
                letter_answer_1 = batch[3][i]
                letter_answer_2 = {0: "A", 1: "B", 2: "C", 3: "D"}[argmax_prob]
                
                ground_truth_text = mmlu_extract_between(full_answer)[letter_answer_1]
                predicted_text = mmlu_extract_text_answer(full_answer)
                
                if (letter_answer_1 in extracted_answer) or \
                    (letter_answer_2 in extracted_answer) or \
                    (ground_truth_text.lower() in predicted_text.lower()):
                    correct += 1
            
            sample_count += batch[0].shape[0]
            
    accumulated_loss /= sample_count
    accuracy = correct / sample_count
    
    return accumulated_loss, accuracy, sample_count

def calibrated_gradients_text_classification_train(net, train_dataloader, test_dataloader):
    if optimizer_name =='adamw': 
        optimizer = AdamW(net.parameters(), lr=lr)
    elif optimizer_name =='adam': 
        optimizer = Adam(net.parameters(), lr=lr)
    elif optimizer_name =='sgd': 
        optimizer = SGD(net.parameters(), lr=lr)
    elif optimizer_name =='sgd_nesterov': 
        optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=True)
    
    losses_epoch = {}
    losses_iter = {}
    accuracies = {}
    times_loss = {}
    times_accuracy = {}
    iteration = 0
    
    # print('Before the training.', flush=True)
    # print('Memory allocated: ', torch.cuda.max_memory_allocated() / (1024 ** 3), flush=True)
    # print('Memory reserved: ', torch.cuda.max_memory_reserved() / (1024 ** 3), flush=True)
    
    params = {name: p for name, p in net.named_parameters() if p.requires_grad}
    trainable_layer_names = list(params.keys())
    
    for epoch in range(1, epochs+1):
        print(f'Epoch {epoch}, Iteration {iteration}')
        net.train()
        # with torch.profiler.profile(
        #     activities=[
        #         torch.profiler.ProfilerActivity.CPU,
        #         torch.profiler.ProfilerActivity.CUDA,
        #     ],
        #     schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True,
        #     on_trace_ready=trace_handler,
        # ) as prof:
        accumulated_loss = 0.0
        start_time = time.time()
        sample_count = 0
        for batch_index, batch in enumerate(train_dataloader):
            # prof.step()
            if len(batch) == 3:
                batch_dict = {
                    'input_ids': batch[0].to(DEVICE),
                    'attention_mask': batch[1].to(DEVICE),
                    'labels': batch[2].to(DEVICE),
                }
            else:
                batch_dict = {
                    'input_ids': batch[0].to(DEVICE),
                    'token_type_ids': batch[1].to(DEVICE),
                    'attention_mask': batch[2].to(DEVICE),
                    'labels': batch[3].to(DEVICE),
                }
                
            # torch.cuda.reset_max_memory_allocated()
            # start_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            # start_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            
            perturbation_seed = random.randint(0, 2**31 - 1)
            
            for perturbation_index in range(perturbation_count): 
                if fixed_seed:
                    torch.manual_seed(perturbation_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(perturbation_seed)
                        torch.cuda.manual_seed_all(perturbation_seed)
    
                if iteration == 0:
                    gradients, tangents, loss, jvp = generate_gradients_for_calibration_phase(
                        net, trainable_layer_names, 
                        params, batch_dict, perturbation_var, jvp_range
                    )
                    
                    accumulated_loss += loss.item()
                    for n, p in params.items():
                        if p.grad == None:
                            p.grad = (gradients[n]).to(p.dtype)
                        else:
                            p.grad += (gradients[n]).to(p.dtype)
                        if torch.isnan(p.grad).any():
                            print('Iteration #: ', iteration, 'NaN Gradient', flush=True)
                        if torch.isnan(p).any():
                            print('Iteration #: ', iteration, 'NaN Weight', flush=True)
                    
                else:
                    tangents = {
                        name: (torch.randn_like(p) * torch.sqrt(torch.tensor(perturbation_var))).to(DEVICE) 
                        for name, p in params.items()
                    }
                
                    with fwAD.dual_level(), torch.no_grad(), torch.cuda.amp.autocast():
                        new_state_dict = {}
                        for n, v in tangents.items():
                            new_state_dict[n] = fwAD.make_dual(params[n], v)
                        net.load_state_dict(new_state_dict, strict=False)
                        del new_state_dict
                        
                        outputs = net(**batch_dict)
                        loss = outputs.loss
                        accumulated_loss += loss.item()
                        
                        jvp = fwAD.unpack_dual(loss).tangent
                        if jvp_range > 0.0:
                            jvp = torch.clamp(jvp, min=-jvp_range, max=jvp_range)
                    
                    if torch.isnan(jvp):
                        print('Iteration #:', iteration, 'Skipping NaN jvp', flush=True)
                        continue
                    
                    for p, v in zip(params.values(), tangents.values()):
                        if p.grad == None:
                            p.grad = (v * jvp).to(p.dtype)
                        else:
                            p.grad += (v * jvp).to(p.dtype)
                        if torch.isnan(p.grad).any():
                            print('Iteration #: ', iteration, 'NaN Gradient', flush=True)
                        if torch.isnan(p).any():
                            print('Iteration #: ', iteration, 'NaN Weight', flush=True)
                
            for p in params.values():
                if p.grad is not None:
                    p.grad /= perturbation_count
            
            wandb.log({
                "loss": loss.item(),
                "jvp": jvp.item(),
                "gradients_mean": {
                    name: param.grad.mean().item() for name, param in net.named_parameters() if param.grad is not None 
                },
                "gradients_std": {
                    name: param.grad.std().item() for name, param in net.named_parameters() if param.grad is not None 
                },
                "gradients_norm": {
                    name: torch.norm(param.grad).item() for name, param in net.named_parameters() if param.grad is not None
                },
                "gradients_max": {
                    name: param.grad.max().item() for name, param in net.named_parameters() if param.grad is not None
                },
                "gradients_min": {
                    name: param.grad.min().item() for name, param in net.named_parameters() if param.grad is not None
                },
                "random_gradient": dict(net.named_parameters())[key_to_track].grad[0][0] if dict(net.named_parameters())[key_to_track].grad is not None else 0,
                "perturbations_mean": {
                    name: pert.mean().item() for name, pert in tangents.items()
                },
                "perturbations_std": {
                    name: pert.std().item() for name, pert in tangents.items()
                },
                "perturbations_max": {
                    name: pert.max().item() for name, pert in tangents.items()
                },
                "perturbations_min": {
                    name: pert.min().item() for name, pert in tangents.items()
                },
                "perturbations_norm": {
                    name: torch.norm(pert).item() for name, pert in tangents.items()
                },
                "random_perturbation": dict(tangents.items())[key_to_track][0][0],
            })
            del tangents
            
            # end_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            # end_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            
            # print('Memory allocated: ', end_memory_allocated - start_memory_allocated, flush=True)    
            # print('Memory reserved: ', end_memory_reserved - start_memory_reserved, flush=True)    
            with record_function("## optimizer ##"):
                optimizer.step()
                optimizer.zero_grad()
                                            
            sample_count += batch_dict['input_ids'].shape[0]
            iteration += 1
            
            losses_iter[iteration] = loss.item() / batch_dict['input_ids'].shape[0] 
            
            wandb.log({
                "parameters_mean": {
                    name: param.data.mean().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_std": {
                    name: param.data.std().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_norm": {
                    name: torch.norm(param.data).item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_max": {
                    name: param.data.max().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_min": {
                    name: param.data.min().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "random_parameter": dict(net.named_parameters())[key_to_track][0][0],
            })
            
            if batch_index >= max_batches: 
                break
                    
        end_time = time.time() 
        print(f'Epoch {epoch} training time: {end_time - start_time}')
        
        start_time = time.time() 
        _, accuracy, _ = text_classification_test(net, test_dataloader)
        end_time = time.time() 
        print(f'Epoch {epoch} test time: {end_time - start_time}')
        
        losses_epoch[epoch] = accumulated_loss / sample_count
        accuracies[epoch] = accuracy
        times_accuracy[end_time - start_time] = accuracy
        times_loss[end_time - start_time] = accumulated_loss / sample_count
        
        print(f'Epoch {epoch} accuracy: {accuracy}')
    
        if epoch % checkpoint_interval == 0:
            with open(results_dir + 'train_losses_epoch.json', 'w+') as f:
                json.dump(losses_epoch, f)
            with open(results_dir + 'train_losses_iter.json', 'w+') as f:
                json.dump(losses_iter, f)
            with open(results_dir + 'accuracies.json', 'w+') as f:
                json.dump(accuracies, f)
            with open(results_dir + 'times_accuracy.json', 'w+') as f:
                json.dump(times_accuracy, f)
            with open(results_dir + 'times_loss.json', 'w+') as f:
                json.dump(times_loss, f)
            
        print('')

def text_classification_train(net, train_dataloader, test_dataloader):
    if optimizer_name =='adamw': 
        optimizer = AdamW(net.parameters(), lr=lr)
    elif optimizer_name =='adam': 
        optimizer = Adam(net.parameters(), lr=lr)
    elif optimizer_name =='sgd': 
        optimizer = SGD(net.parameters(), lr=lr)
    elif optimizer_name =='sgd_nesterov': 
        optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=True)
    
    losses_epoch = {}
    losses_iter = {}
    accuracies = {}
    times_loss = {}
    times_accuracy = {}
    iteration = 0
    
    # print('Before the training.', flush=True)
    # print('Memory allocated: ', torch.cuda.max_memory_allocated() / (1024 ** 3), flush=True)
    # print('Memory reserved: ', torch.cuda.max_memory_reserved() / (1024 ** 3), flush=True)
    
    params = {name: p for name, p in net.named_parameters() if p.requires_grad}
    
    for epoch in range(1, epochs+1):
        net.train()
        # with torch.profiler.profile(
        #     activities=[
        #         torch.profiler.ProfilerActivity.CPU,
        #         torch.profiler.ProfilerActivity.CUDA,
        #     ],
        #     schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True,
        #     on_trace_ready=trace_handler,
        # ) as prof:
        accumulated_loss = 0.0
        start_time = time.time()
        sample_count = 0
        for batch_index, batch in enumerate(train_dataloader):
            # prof.step()
            if len(batch) == 3:
                batch_dict = {
                    'input_ids': batch[0].to(DEVICE),
                    'attention_mask': batch[1].to(DEVICE),
                    'labels': batch[2].to(DEVICE),
                    'use_cache': False,
                }
            else:
                batch_dict = {
                    'input_ids': batch[0].to(DEVICE),
                    'token_type_ids': batch[1].to(DEVICE),
                    'attention_mask': batch[2].to(DEVICE),
                    'labels': batch[3].to(DEVICE),
                    'use_cache': False,
                }
                
            # torch.cuda.reset_max_memory_allocated()
            # start_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            # start_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            
            perturbation_seed = random.randint(0, 2**31 - 1)
            
            for perturbation_index in range(perturbation_count): 
                if fixed_seed:
                    torch.manual_seed(perturbation_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(perturbation_seed)
                        torch.cuda.manual_seed_all(perturbation_seed)
    
                tangents = {
                    name: (torch.randn_like(p) * torch.sqrt(torch.tensor(perturbation_var))).to(DEVICE) 
                    for name, p in params.items()
                }
                
                with fwAD.dual_level(), torch.no_grad(), torch.cuda.amp.autocast():
                    new_state_dict = {}
                    for n, p in params.items():
                        new_state_dict[n] = fwAD.make_dual(p, tangents[n])
                    net.load_state_dict(new_state_dict, strict=False)
                    del new_state_dict
                    
                    for n, p in net.named_parameters():
                        if p.requires_grad:
                            if torch.isnan(p).any():
                                print(n, ' contains NaN')
                    
                    outputs = net(**batch_dict)
                    
                    loss = outputs.loss
                    accumulated_loss += loss.item()
                    
                    jvp = fwAD.unpack_dual(loss).tangent
                    jvp = torch.clamp(jvp, min=-jvp_range, max=jvp_range)
                
                if torch.isnan(jvp):
                    print('Iteration #:', iteration, 'Skipping NaN jvp', flush=True)
                    continue
                
                for p, v in zip(params.values(), tangents.values()):
                    if p.grad == None:
                        p.grad = v * jvp
                    else:
                        p.grad += v * jvp
                
                del tangents
            
            for p in params.values():
                if p.grad is not None:
                    # p.grad[(p.grad == 0)] = 1e-6
                    p.grad /= perturbation_count
            
            wandb.log({
                "loss": loss.item(),
                "jvp": jvp.item(),
                "gradients_mean": {
                    name: param.grad.mean().item() for name, param in net.named_parameters() if param.grad is not None 
                },
                "gradients_std": {
                    name: param.grad.std().item() for name, param in net.named_parameters() if param.grad is not None 
                },
                "gradients_norm": {
                    name: torch.norm(param.grad).item() for name, param in net.named_parameters() if param.grad is not None
                },
                "gradients_max": {
                    name: param.grad.max().item() for name, param in net.named_parameters() if param.grad is not None
                },
                "gradients_min": {
                    name: param.grad.min().item() for name, param in net.named_parameters() if param.grad is not None
                },
            })
            
            # end_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            # end_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            
            # print('Memory allocated: ', end_memory_allocated - start_memory_allocated, flush=True)    
            # print('Memory reserved: ', end_memory_reserved - start_memory_reserved, flush=True)    
            with record_function("## optimizer ##"):
                optimizer.step()
                optimizer.zero_grad()
                                            
            sample_count += batch_dict['input_ids'].shape[0]
            iteration += 1
            
            losses_iter[iteration] = loss.item() / batch_dict['input_ids'].shape[0] 
            
            wandb.log({
                "parameters_mean": {
                    name: param.data.mean().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_std": {
                    name: param.data.std().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_norm": {
                    name: torch.norm(param.data).item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_max": {
                    name: param.data.max().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_min": {
                    name: param.data.min().item() for name, param in net.named_parameters() if param.requires_grad
                },
            })
            
            if batch_index >= max_batches: 
                break
                    
        end_time = time.time() 
        print(f'Epoch {epoch} training time: {end_time - start_time}')
        
        start_test_time = time.time() 
        _, accuracy, _ = text_classification_test(net, test_dataloader)
        end_test_time = time.time() 
        print(f'Epoch {epoch} test time: {end_test_time - start_test_time}')
        
        accuracies[epoch] = accuracy
        losses_epoch[epoch] = accumulated_loss / sample_count
        times_accuracy[end_time - start_time] = accuracy
        times_loss[end_time - start_time] = accumulated_loss / sample_count
        
        print(f'Epoch {epoch} accuracy: {accuracy}')
    
        if epoch % checkpoint_interval == 0:
            with open(results_dir + 'train_losses_epoch.json', 'w+') as f:
                json.dump(losses_epoch, f)
            with open(results_dir + 'train_losses_iter.json', 'w+') as f:
                json.dump(losses_iter, f)
            with open(results_dir + 'accuracies.json', 'w+') as f:
                json.dump(accuracies, f)
            with open(results_dir + 'times_accuracy.json', 'w+') as f:
                json.dump(times_accuracy, f)
            with open(results_dir + 'times_loss.json', 'w+') as f:
                json.dump(times_loss, f)
            
        print('')
           
def gradient_accumulation_text_classification_train(net, train_dataloader, test_dataloader):
    if optimizer_name =='adamw': 
        optimizer = AdamW(net.parameters(), lr=lr)
    elif optimizer_name =='adam': 
        optimizer = Adam(net.parameters(), lr=lr)
    elif optimizer_name =='sgd': 
        optimizer = SGD(net.parameters(), lr=lr)
    elif optimizer_name =='sgd_nesterov': 
        optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=True)
    
    losses_epoch = {}
    losses_iter = {}
    accuracies = {}
    times_loss = {}
    times_accuracy = {}
    
    iteration = 0
    valid_iteration = 0
    
    # print('Before the training.', flush=True)
    # print('Memory allocated: ', torch.cuda.max_memory_allocated() / (1024 ** 3), flush=True)
    # print('Memory reserved: ', torch.cuda.max_memory_reserved() / (1024 ** 3), flush=True)
    
    params = {name: p for name, p in net.named_parameters() if p.requires_grad}
    
    for epoch in range(1, epochs+1):
        print(f'Epoch {epoch}, Iteration {iteration}')
        net.train()
        # with torch.profiler.profile(
        #     activities=[
        #         torch.profiler.ProfilerActivity.CPU,
        #         torch.profiler.ProfilerActivity.CUDA,
        #     ],
        #     schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True,
        #     on_trace_ready=trace_handler,
        # ) as prof:
        accumulated_loss = 0.0
        start_time = time.time()
        sample_count = 0
        for batch_index, batch in enumerate(train_dataloader):
            # print("batch_index", batch_index, end='\r') 
            # prof.step()
            if len(batch) == 3:
                batch_dict = {
                    'input_ids': batch[0].to(DEVICE),
                    'attention_mask': batch[1].to(DEVICE),
                    'labels': batch[2].to(DEVICE),
                    'use_cache': False,
                }
            else:
                batch_dict = {
                    'input_ids': batch[0].to(DEVICE),
                    'token_type_ids': batch[1].to(DEVICE),
                    'attention_mask': batch[2].to(DEVICE),
                    'labels': batch[3].to(DEVICE),
                    'use_cache': False,
                }
                
            # torch.cuda.reset_max_memory_allocated()
            # start_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            # start_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            
            if fixed_seed:
                perturbation_seed = random.randint(0, 2**31 - 1)
                torch.manual_seed(perturbation_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(perturbation_seed)
                    torch.cuda.manual_seed_all(perturbation_seed)

            tangents = {
                name: (torch.randn_like(p) * torch.sqrt(torch.tensor(perturbation_var))).to(DEVICE) 
                for name, p in params.items()
            }
            
            ## Uniform distribution
            # tangents = {
            #     name: (torch.empty(p.shape, dtype=p.dtype).uniform_(-1e-4, 1e-4)).to(DEVICE) 
            #     for name, p in params.items()
            # }
            
            with fwAD.dual_level(), torch.no_grad():# torch.cuda.amp.autocast(), record_function("## forward ##"):
                new_state_dict = {}
                for n, p in params.items():
                    new_state_dict[n] = fwAD.make_dual(p, tangents[n])
                net.load_state_dict(new_state_dict, strict=False)
                del new_state_dict
                
                outputs = net(**batch_dict)
                loss = outputs.loss
                accumulated_loss += loss.item()
                
                jvp = fwAD.unpack_dual(loss).tangent
                if jvp_range > 0.0:
                    jvp = torch.clamp(jvp, min=-jvp_range, max=jvp_range)
            
            if torch.isnan(jvp):
                print('Iteration #:', iteration, 'Skipping NaN jvp', flush=True)
                continue
            else:
                valid_iteration += 1
            
            for p, v in zip(params.values(), tangents.values()):
                if p.grad == None:
                    p.grad = v * jvp
                else:
                    p.grad += v * jvp
            
            if valid_iteration % accumulation_steps == 0:
                print('Accumulating gradients at ', iteration, flush=True)
                for p in params.values():
                    p.grad /= accumulation_steps
            
                if grad_norm_range > 0.0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_norm_range)
                if grad_value_range > 0.0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_value_range)
            
                wandb.log({
                    "loss": loss.item(),
                    "jvp": jvp.item(),
                    "gradients_mean": {
                        name: param.grad.mean().item() for name, param in net.named_parameters() if param.grad is not None 
                    },
                    "gradients_std": {
                        name: param.grad.std().item() for name, param in net.named_parameters() if param.grad is not None 
                    },
                    "gradients_norm": {
                        name: torch.norm(param.grad).item() for name, param in net.named_parameters() if param.grad is not None
                    },
                    "gradients_max": {
                        name: param.grad.max().item() for name, param in net.named_parameters() if param.grad is not None
                    },
                    "gradients_min": {
                        name: param.grad.min().item() for name, param in net.named_parameters() if param.grad is not None
                    },
                    "random_gradient": dict(net.named_parameters())[key_to_track].grad[0][0] if dict(net.named_parameters())[key_to_track].grad is not None else 0,
                    "perturbations_mean": {
                        name: pert.mean().item() for name, pert in tangents.items()
                    },
                    "perturbations_std": {
                        name: pert.std().item() for name, pert in tangents.items()
                    },
                    "perturbations_max": {
                        name: pert.max().item() for name, pert in tangents.items()
                    },
                    "perturbations_min": {
                        name: pert.min().item() for name, pert in tangents.items()
                    },
                    "perturbations_norm": {
                        name: torch.norm(pert).item() for name, pert in tangents.items()
                    },
                    "random_perturbation": dict(tangents.items())[key_to_track][0][0],
                })
                del tangents
            
                # end_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
                # end_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
                
                # print('Memory allocated: ', end_memory_allocated - start_memory_allocated, flush=True)    
                # print('Memory reserved: ', end_memory_reserved - start_memory_reserved, flush=True)    
                with record_function("## optimizer ##"):
                    optimizer.step()
                    optimizer.zero_grad()
                                            
            sample_count += batch_dict['input_ids'].shape[0]
            iteration += 1
            
            losses_iter[iteration] = loss.item() / batch_dict['input_ids'].shape[0] 
            
            wandb.log({
                "parameters_mean": {
                    name: param.data.mean().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_std": {
                    name: param.data.std().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_norm": {
                    name: torch.norm(param.data).item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_max": {
                    name: param.data.max().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_min": {
                    name: param.data.min().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "random_parameter": dict(net.named_parameters())[key_to_track][0][0],
            })
            
            if batch_index >= max_batches: 
                break
                    
        end_time = time.time() 
        print(f'Epoch {epoch} training time: {end_time - start_time}')
           
        start_test_time = time.time() 
        _, accuracy, _ = text_classification_test(net, test_dataloader)
        end_test_time = time.time() 
        print(f'Epoch {epoch} test time: {end_test_time - start_test_time}')
        
        losses_epoch[epoch] = accumulated_loss / sample_count
        accuracies[epoch] = accuracy
        times_accuracy[end_time - start_time] = accuracy
        times_loss[end_time - start_time] = accumulated_loss / sample_count
        
        print(f'Epoch {epoch} accuracy: {accuracy}')
    
        if epoch % checkpoint_interval == 0:
            with open(results_dir + 'train_losses_epoch.json', 'w+') as f:
                json.dump(losses_epoch, f)
            with open(results_dir + 'train_losses_iter.json', 'w+') as f:
                json.dump(losses_iter, f)
            with open(results_dir + 'accuracies.json', 'w+') as f:
                json.dump(accuracies, f)
            with open(results_dir + 'times_accuracy.json', 'w+') as f:
                json.dump(times_accuracy, f)
            with open(results_dir + 'times_loss.json', 'w+') as f:
                json.dump(times_loss, f)
            
        print('')
            
def calibrated_gradient_accumulation_text_classification_train(net, train_dataloader, test_dataloader):
    if optimizer_name =='adamw': 
        optimizer = AdamW(net.parameters(), lr=lr)
    elif optimizer_name =='adam': 
        optimizer = Adam(net.parameters(), lr=lr)
    elif optimizer_name =='sgd': 
        optimizer = SGD(net.parameters(), lr=lr)
    elif optimizer_name =='sgd_nesterov': 
        optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=True)
    
    losses_epoch = {}
    losses_iter = {}
    accuracies = {}
    times_loss = {}
    times_accuracy = {}
    
    iteration = 0
    valid_iteration = 0
    
    # print('Before the training.', flush=True)
    # print('Memory allocated: ', torch.cuda.max_memory_allocated() / (1024 ** 3), flush=True)
    # print('Memory reserved: ', torch.cuda.max_memory_reserved() / (1024 ** 3), flush=True)
    params = {name: p for name, p in net.named_parameters() if p.requires_grad}
    trainable_layer_names = list(params.keys())
    
    for batch_index, batch in enumerate(train_dataloader):
        # prof.step()
        if len(batch) == 3:
            batch_dict = {
                'input_ids': batch[0].to(DEVICE),
                'attention_mask': batch[1].to(DEVICE),
                'labels': batch[2].to(DEVICE),
            }
        else:
            batch_dict = {
                'input_ids': batch[0].to(DEVICE),
                'token_type_ids': batch[1].to(DEVICE),
                'attention_mask': batch[2].to(DEVICE),
                'labels': batch[3].to(DEVICE),
            }
        
        gradients = generate_gradients_for_calibration_phase(
            net, trainable_layer_names, 
            params, batch_dict, perturbation_var, jvp_range
        )
            
        for n, p in params.items():
            if p.grad == None:
                p.grad = gradients[n]
            else:
                p.grad += gradients[n]
        
        iteration += 1        
        if batch_index == accumulation_steps - 1:
            break
    
    for n, p in params.items():
        if p.grad is not None:
            p.grad /= accumulation_steps
                
    optimizer.step()
    optimizer.zero_grad()
    
    for epoch in range(1, epochs+1):
        print(f'Epoch {epoch}, Iteration {iteration}')
        net.train()
        # with torch.profiler.profile(
        #     activities=[
        #         torch.profiler.ProfilerActivity.CPU,
        #         torch.profiler.ProfilerActivity.CUDA,
        #     ],
        #     schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True,
        #     on_trace_ready=trace_handler,
        # ) as prof:
        accumulated_loss = 0.0
        start_time = time.time()
        sample_count = 0
        for batch_index, batch in enumerate(train_dataloader):
            # prof.step()
            if len(batch) == 3:
                batch_dict = {
                    'input_ids': batch[0].to(DEVICE),
                    'attention_mask': batch[1].to(DEVICE),
                    'labels': batch[2].to(DEVICE),
                }
            else:
                batch_dict = {
                    'input_ids': batch[0].to(DEVICE),
                    'token_type_ids': batch[1].to(DEVICE),
                    'attention_mask': batch[2].to(DEVICE),
                    'labels': batch[3].to(DEVICE),
                }
                
            # torch.cuda.reset_max_memory_allocated()
            # start_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            # start_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            
            perturbation_seed = random.randint(0, 2**31 - 1)
            
            if fixed_seed:
                torch.manual_seed(perturbation_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(perturbation_seed)
                    torch.cuda.manual_seed_all(perturbation_seed)

            tangents = {
                name: (torch.randn_like(p) * torch.sqrt(torch.tensor(perturbation_var))).to(DEVICE) 
                for name, p in params.items()
            }
            
            with fwAD.dual_level(), torch.no_grad(), record_function("## forward ##"):
                new_state_dict = {}
                for n, p in params.items():
                    new_state_dict[n] = fwAD.make_dual(p, tangents[n])
                net.load_state_dict(new_state_dict, strict=False)
                del new_state_dict
                
                outputs = net(**batch_dict)
                loss = outputs.loss
                accumulated_loss += loss.item()
                
                jvp = fwAD.unpack_dual(loss).tangent
                if jvp_range > 0.0:
                    jvp = torch.clamp(jvp, min=-jvp_range, max=jvp_range)
            
            if torch.isnan(jvp):
                print('Iteration #:', iteration, 'Skipping NaN jvp', flush=True)
                continue
            else:
                valid_iteration += 1
            
            for p, v in zip(params.values(), tangents.values()):
                if p.grad == None:
                    p.grad = v * jvp
                else:
                    p.grad += v * jvp
            
            if valid_iteration % accumulation_steps == 0:
                print('Accumulating gradients at ', iteration, flush=True)
                for p in params.values():
                    if p.grad is not None:
                        p.grad /= accumulation_steps
            
                if grad_norm_range > 0.0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_norm_range)
                if grad_value_range > 0.0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_value_range)
            
                wandb.log({
                    "loss": loss.item(),
                    "jvp": jvp.item(),
                    "gradients_mean": {
                        name: param.grad.mean().item() for name, param in net.named_parameters() if param.grad is not None 
                    },
                    "gradients_std": {
                        name: param.grad.std().item() for name, param in net.named_parameters() if param.grad is not None 
                    },
                    "gradients_norm": {
                        name: torch.norm(param.grad).item() for name, param in net.named_parameters() if param.grad is not None
                    },
                    "gradients_max": {
                        name: param.grad.max().item() for name, param in net.named_parameters() if param.grad is not None
                    },
                    "gradients_min": {
                        name: param.grad.min().item() for name, param in net.named_parameters() if param.grad is not None
                    },
                    "random_gradient": dict(net.named_parameters())[key_to_track].grad[0][0] if dict(net.named_parameters())[key_to_track].grad is not None else 0,
                    "perturbations_mean": {
                        name: pert.mean().item() for name, pert in tangents.items()
                    },
                    "perturbations_std": {
                        name: pert.std().item() for name, pert in tangents.items()
                    },
                    "perturbations_max": {
                        name: pert.max().item() for name, pert in tangents.items()
                    },
                    "perturbations_min": {
                        name: pert.min().item() for name, pert in tangents.items()
                    },
                    "perturbations_norm": {
                        name: torch.norm(pert).item() for name, pert in tangents.items()
                    },
                    "random_perturbation": dict(tangents.items())[key_to_track][0][0],
                })
                del tangents
            
                # end_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
                # end_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
                
                # print('Memory allocated: ', end_memory_allocated - start_memory_allocated, flush=True)    
                # print('Memory reserved: ', end_memory_reserved - start_memory_reserved, flush=True)    
                with record_function("## optimizer ##"):
                    optimizer.step()
                    optimizer.zero_grad()
                                            
            sample_count += batch_dict['input_ids'].shape[0]
            iteration += 1
            
            losses_iter[iteration] = loss.item() / batch_dict['input_ids'].shape[0] 
            
            wandb.log({
                "parameters_mean": {
                    name: param.data.mean().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_std": {
                    name: param.data.std().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_norm": {
                    name: torch.norm(param.data).item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_max": {
                    name: param.data.max().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_min": {
                    name: param.data.min().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "random_parameter": dict(net.named_parameters())[key_to_track][0][0],
            })
            
            if batch_index >= max_batches: 
                break
                    
        end_time = time.time() 
        print(f'Epoch {epoch} training time: {end_time - start_time}')
           
        start_test_time = time.time() 
        _, accuracy, _ = text_classification_test(net, test_dataloader)
        end_test_time = time.time() 
        print(f'Epoch {epoch} test time: {end_test_time - start_test_time}')
        
        losses_epoch[epoch] = accumulated_loss / sample_count
        accuracies[epoch] = accuracy
        times_accuracy[end_time - start_time] = accuracy
        times_loss[end_time - start_time] = accumulated_loss / sample_count
        
        print(f'Epoch {epoch} accuracy: {accuracy}')
    
        if epoch % checkpoint_interval == 0:
            with open(results_dir + 'train_losses_epoch.json', 'w+') as f:
                json.dump(losses_epoch, f)
            with open(results_dir + 'train_losses_iter.json', 'w+') as f:
                json.dump(losses_iter, f)
            with open(results_dir + 'accuracies.json', 'w+') as f:
                json.dump(accuracies, f)
            with open(results_dir + 'times_accuracy.json', 'w+') as f:
                json.dump(times_accuracy, f)
            with open(results_dir + 'times_loss.json', 'w+') as f:
                json.dump(times_loss, f)
            
        print('')
            
def text_classification_test(net, test_dataloader):
    metric = load_metric("accuracy")
    
    correct = 0
    accumulated_loss = 0
    sample_count = 0
    net.eval()
    
    for batch in test_dataloader:
        if len(batch) == 3:
            batch_dict = {
                'input_ids': batch[0].to(DEVICE),
                'attention_mask': batch[1].to(DEVICE),
                'labels': batch[2].to(DEVICE),
            }
        else:
            batch_dict = {
                'input_ids': batch[0].to(DEVICE),
                'token_type_ids': batch[1].to(DEVICE),
                'attention_mask': batch[2].to(DEVICE),
                'labels': batch[3].to(DEVICE),
            }
        
        try:
            with torch.no_grad():
                outputs = net(**batch_dict)
                
            logits = outputs.logits
            accumulated_loss += outputs.loss.item()
            sample_count += batch[0].shape[0]
            
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch_dict["labels"])
        except Exception as e:
            print(e)
            
    accumulated_loss /= sample_count
    accuracy = metric.compute()["accuracy"]
    print('Test accuracy: ', accuracy)
    
    return accumulated_loss, accuracy, sample_count

def calibrated_gradients_vqa_train(net, train_dataloader, test_dataloader):
    if optimizer_name =='adamw': 
        optimizer = AdamW(net.parameters(), lr=lr)
    elif optimizer_name =='adam': 
        optimizer = Adam(net.parameters(), lr=lr)
    elif optimizer_name =='sgd': 
        optimizer = SGD(net.parameters(), lr=lr)
    elif optimizer_name =='sgd_nesterov': 
        optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=True)
    
    losses_epoch = {}
    losses_iter = {}
    accuracies = {}
    times_loss = {}
    times_accuracy = {}
    iteration = 0
    
    # print('Before the training.', flush=True)
    # print('Memory allocated: ', torch.cuda.max_memory_allocated() / (1024 ** 3), flush=True)
    # print('Memory reserved: ', torch.cuda.max_memory_reserved() / (1024 ** 3), flush=True)
    
    params = {name: p for name, p in net.named_parameters() if p.requires_grad}
    trainable_layer_names = list(params.keys())
    
    for epoch in range(1, epochs+1):
        print(f'Epoch {epoch}, Iteration {iteration}')
        net.train()
        # with torch.profiler.profile(
        #     activities=[
        #         torch.profiler.ProfilerActivity.CPU,
        #         torch.profiler.ProfilerActivity.CUDA,
        #     ],
        #     schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True,
        #     on_trace_ready=trace_handler,
        # ) as prof:
        accumulated_loss = 0.0
        start_time = time.time()
        sample_count = 0
        for batch_index, batch in enumerate(train_dataloader):
            # prof.step()
            batch_dict = {
                'input_ids': batch['input_ids'].to(DEVICE),
                'pixel_values': batch['pixel_values'].to(DEVICE),
                'attention_mask': batch['attention_mask'].to(DEVICE),
                # 'labels': batch['labels'].to(DEVICE),
                'image_grid_thw': batch['image_grid_thw'].to(DEVICE),
                'labels': batch['input_ids'].to(DEVICE),
            }
                
            # torch.cuda.reset_max_memory_allocated()
            # start_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            # start_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            
            perturbation_seed = random.randint(0, 2**31 - 1)
            
            for perturbation_index in range(perturbation_count): 
                if fixed_seed:
                    torch.manual_seed(perturbation_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(perturbation_seed)
                        torch.cuda.manual_seed_all(perturbation_seed)
    
                if iteration == 0:
                    gradients, tangents, loss, jvp = generate_gradients_for_calibration_phase(
                        net, trainable_layer_names, 
                        params, batch_dict, perturbation_var, jvp_range
                    )
                    
                    accumulated_loss += loss.item()
                    for n, p in params.items():
                        if p.grad == None:
                            p.grad = (gradients[n]).to(p.dtype)
                        else:
                            p.grad += (gradients[n]).to(p.dtype)
                        if torch.isnan(p.grad).any():
                            print('Iteration #: ', iteration, 'NaN Gradient', flush=True)
                        if torch.isnan(p).any():
                            print('Iteration #: ', iteration, 'NaN Weight', flush=True)
                    
                else:
                    tangents = {
                        name: (torch.randn_like(p) * torch.sqrt(torch.tensor(perturbation_var))).to(DEVICE) 
                        for name, p in params.items()
                    }
                
                    with fwAD.dual_level(), torch.no_grad(), torch.cuda.amp.autocast():
                        new_state_dict = {}
                        for n, v in tangents.items():
                            new_state_dict[n] = fwAD.make_dual(params[n], v)
                        net.load_state_dict(new_state_dict, strict=False)
                        del new_state_dict
                        
                        outputs = net(**batch_dict)
                        loss = outputs.loss
                        accumulated_loss += loss.item()
                        
                        jvp = fwAD.unpack_dual(loss).tangent
                        if jvp_range > 0.0:
                            jvp = torch.clamp(jvp, min=-jvp_range, max=jvp_range)
                    
                    if torch.isnan(jvp):
                        print('Iteration #:', iteration, 'Skipping NaN jvp', flush=True)
                        continue
                    
                    for p, v in zip(params.values(), tangents.values()):
                        if p.grad == None:
                            p.grad = (v * jvp).to(p.dtype)
                        else:
                            p.grad += (v * jvp).to(p.dtype)
                        if torch.isnan(p.grad).any():
                            print('Iteration #: ', iteration, 'NaN Gradient', flush=True)
                        if torch.isnan(p).any():
                            print('Iteration #: ', iteration, 'NaN Weight', flush=True)
                
            for p in params.values():
                if p.grad is not None:
                    p.grad /= perturbation_count
            
            wandb.log({
                "loss": loss.item(),
                "jvp": jvp.item(),
                "gradients_mean": {
                    name: param.grad.mean().item() for name, param in net.named_parameters() if param.grad is not None 
                },
                "gradients_std": {
                    name: param.grad.std().item() for name, param in net.named_parameters() if param.grad is not None 
                },
                "gradients_norm": {
                    name: torch.norm(param.grad).item() for name, param in net.named_parameters() if param.grad is not None
                },
                "gradients_max": {
                    name: param.grad.max().item() for name, param in net.named_parameters() if param.grad is not None
                },
                "gradients_min": {
                    name: param.grad.min().item() for name, param in net.named_parameters() if param.grad is not None
                },
                "random_gradient": dict(net.named_parameters())[key_to_track].grad[0][0] if dict(net.named_parameters())[key_to_track].grad is not None else 0,
                "perturbations_mean": {
                    name: pert.mean().item() for name, pert in tangents.items()
                },
                "perturbations_std": {
                    name: pert.std().item() for name, pert in tangents.items()
                },
                "perturbations_max": {
                    name: pert.max().item() for name, pert in tangents.items()
                },
                "perturbations_min": {
                    name: pert.min().item() for name, pert in tangents.items()
                },
                "perturbations_norm": {
                    name: torch.norm(pert).item() for name, pert in tangents.items()
                },
                "random_perturbation": dict(tangents.items())[key_to_track][0][0],
            })
            del tangents
            
            # end_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            # end_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            
            # print('Memory allocated: ', end_memory_allocated - start_memory_allocated, flush=True)    
            # print('Memory reserved: ', end_memory_reserved - start_memory_reserved, flush=True)    
            with record_function("## optimizer ##"):
                optimizer.step()
                optimizer.zero_grad()
                                            
            sample_count += batch_dict['input_ids'].shape[0]
            iteration += 1
            
            losses_iter[iteration] = loss.item() / batch_dict['input_ids'].shape[0] 
            
            wandb.log({
                "parameters_mean": {
                    name: param.data.mean().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_std": {
                    name: param.data.std().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_norm": {
                    name: torch.norm(param.data).item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_max": {
                    name: param.data.max().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_min": {
                    name: param.data.min().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "random_parameter": dict(net.named_parameters())[key_to_track][0][0],
            })
            
            if batch_index >= max_batches: 
                break
                    
        end_time = time.time() 
        print(f'Epoch {epoch} training time: {end_time - start_time}')
        
        start_time = time.time() 
        _, accuracy, _ = vqa_test(net, test_dataloader)
        end_time = time.time() 
        print(f'Epoch {epoch} test time: {end_time - start_time}')
        
        losses_epoch[epoch] = accumulated_loss / sample_count
        accuracies[epoch] = accuracy
        times_accuracy[end_time - start_time] = accuracy
        times_loss[end_time - start_time] = accumulated_loss / sample_count
        
        print(f'Epoch {epoch} accuracy: {accuracy}')
    
        if epoch % checkpoint_interval == 0:
            with open(results_dir + 'train_losses_epoch.json', 'w+') as f:
                json.dump(losses_epoch, f)
            with open(results_dir + 'train_losses_iter.json', 'w+') as f:
                json.dump(losses_iter, f)
            with open(results_dir + 'accuracies.json', 'w+') as f:
                json.dump(accuracies, f)
            with open(results_dir + 'times_accuracy.json', 'w+') as f:
                json.dump(times_accuracy, f)
            with open(results_dir + 'times_loss.json', 'w+') as f:
                json.dump(times_loss, f)
            
        print('')

def gradient_accumulation_vqa_train(net, train_dataloader, test_dataloader):
    if optimizer_name =='adamw': 
        optimizer = AdamW(net.parameters(), lr=lr)
    elif optimizer_name =='adam': 
        optimizer = Adam(net.parameters(), lr=lr)
    elif optimizer_name =='nadam': 
        optimizer = NAdam(net.parameters(), lr=lr, momentum_decay=momentum)
    elif optimizer_name =='sgd': 
        optimizer = SGD(net.parameters(), lr=lr)
    elif optimizer_name =='sgd_nesterov': 
        optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=True)
    
    losses_epoch = {}
    losses_iter = {}
    accuracies = {}
    times_loss = {}
    times_accuracy = {}
    iteration = 0
    
    valid_iteration = 0
    
    # print('Before the training.', flush=True)
    # print('Memory allocated: ', torch.cuda.max_memory_allocated() / (1024 ** 3), flush=True)
    # print('Memory reserved: ', torch.cuda.max_memory_reserved() / (1024 ** 3), flush=True)
    
    params = {name: p for name, p in net.named_parameters() if p.requires_grad}
    
    for epoch in range(1, epochs+1):
        print(f'Epoch {epoch}, Iteration {iteration}')
        net.train()
        # with torch.profiler.profile(
        #     activities=[
        #         torch.profiler.ProfilerActivity.CPU,
        #         torch.profiler.ProfilerActivity.CUDA,
        #     ],
        #     # schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True,
        #     on_trace_ready=trace_handler,
        # ) as prof:
        accumulated_loss = 0.0
        start_time = time.time()
        sample_count = 0
        for batch_index, batch in enumerate(train_dataloader):
            # prof.step()
            batch_dict = {
                'input_ids': batch['input_ids'].to(DEVICE),
                'pixel_values': batch['pixel_values'].to(DEVICE),
                'attention_mask': batch['attention_mask'].to(DEVICE),
                # 'labels': batch['labels'].to(DEVICE),
                'image_grid_thw': batch['image_grid_thw'].to(DEVICE),
                'labels': batch['input_ids'].to(DEVICE),
                'use_cache': False,
            }
            
            # torch.cuda.reset_max_memory_allocated()
            # start_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            # start_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            
            if fixed_seed:
                perturbation_seed = random.randint(0, 2**31 - 1)
                torch.manual_seed(perturbation_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(perturbation_seed)
                    torch.cuda.manual_seed_all(perturbation_seed)

            tangents = {
                name: (torch.randn_like(p) * torch.sqrt(torch.tensor(perturbation_var))).to(DEVICE) 
                for name, p in params.items()
            }
                
            with fwAD.dual_level(), torch.no_grad(), record_function("## forward ##"):
                new_state_dict = {}
                for n, p in params.items():
                    new_state_dict[n] = fwAD.make_dual(p, tangents[n])
                net.load_state_dict(new_state_dict, strict=False)
                del new_state_dict
                
                outputs = net(**batch_dict)
                loss = outputs.loss
                accumulated_loss += loss.item()
                
                jvp = fwAD.unpack_dual(loss).tangent
                if jvp_range > 0.0:
                    jvp = torch.clamp(jvp, min=-jvp_range, max=jvp_range)
                
                if torch.isnan(jvp):
                    print('Iteration #:', iteration, 'Skipping NaN jvp', flush=True)
                    continue
                else:
                    valid_iteration += 1
                
            for p, v in zip(params.values(), tangents.values()):
                if p.grad == None:
                    p.grad = v * jvp
                else:
                    p.grad += v * jvp
                if torch.isnan(p.grad).any():
                    print('Iteration #: ', iteration, 'NaN Gradient', flush=True)
                if torch.isnan(p).any():
                    print('Iteration #: ', iteration, 'NaN Weight', flush=True)
                 
            if valid_iteration % accumulation_steps == 0:       
                for p in params.values():
                    p.grad /= accumulation_steps
                        
                if grad_norm_range > 0.0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_norm_range)
                if grad_value_range > 0.0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_value_range)
            
                wandb.log({
                    "loss": loss.item(),
                    "jvp": jvp.item(),
                    "gradients_mean": {
                        name: param.grad.mean().item() for name, param in net.named_parameters() if param.grad is not None 
                    },
                    "gradients_std": {
                        name: param.grad.std().item() for name, param in net.named_parameters() if param.grad is not None 
                    },
                    "gradients_norm": {
                        name: torch.norm(param.grad).item() for name, param in net.named_parameters() if param.grad is not None
                    },
                    "gradients_max": {
                        name: param.grad.max().item() for name, param in net.named_parameters() if param.grad is not None
                    },
                    "gradients_min": {
                        name: param.grad.min().item() for name, param in net.named_parameters() if param.grad is not None
                    },
                    "random_gradient": dict(net.named_parameters())[key_to_track].grad[0][0] if dict(net.named_parameters())[key_to_track].grad is not None else 0,
                    "perturbations_mean": {
                        name: pert.mean().item() for name, pert in tangents.items()
                    },
                    "perturbations_std": {
                        name: pert.std().item() for name, pert in tangents.items()
                    },
                    "perturbations_max": {
                        name: pert.max().item() for name, pert in tangents.items()
                    },
                    "perturbations_min": {
                        name: pert.min().item() for name, pert in tangents.items()
                    },
                    "perturbations_norm": {
                        name: torch.norm(pert).item() for name, pert in tangents.items()
                    },
                    "random_perturbation": dict(tangents.items())[key_to_track][0][0],
                })
                del tangents
            
                # end_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
                # end_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
                
                # print('Memory allocated: ', end_memory_allocated - start_memory_allocated, flush=True)    
                # print('Memory reserved: ', end_memory_reserved - start_memory_reserved, flush=True)    
                with record_function("## optimizer ##"):
                    optimizer.step()
                    
                    if optimizer_name =='adamw' or optimizer_name =='adam':
                        wandb.log({
                            "exp_avg_mean": {
                                name: optimizer.state[param]['exp_avg'].mean().item() for name, param in params.items()
                            },
                            "exp_avg_max": {
                                name: optimizer.state[param]['exp_avg'].max().item() for name, param in params.items()
                            },
                            "exp_avg_min": {
                                name: optimizer.state[param]['exp_avg'].min().item() for name, param in params.items()
                            },
                            "exp_avg_sq_mean": {
                                name: optimizer.state[param]['exp_avg_sq'].mean().item() for name, param in params.items()
                            },
                            "exp_avg_sq_max": {
                                name: optimizer.state[param]['exp_avg_sq'].max().item() for name, param in params.items()
                            },
                            "exp_avg_sq_min": {
                                name: optimizer.state[param]['exp_avg_sq'].min().item() for name, param in params.items()
                            },
                            "random_parameter_avg_sq": optimizer.state[params[key_to_track]]['exp_avg_sq'][0][0],
                            "random_parameter_avg": optimizer.state[params[key_to_track]]['exp_avg'][0][0],
                        })
                        
                    optimizer.zero_grad()
                
            sample_count += batch_dict['input_ids'].shape[0]
            iteration += 1
            
            losses_iter[iteration] = loss.item() / batch_dict['input_ids'].shape[0] 
            
            wandb.log({
                "parameters_mean": {
                    name: param.data.mean().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_std": {
                    name: param.data.std().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_norm": {
                    name: torch.norm(param.data).item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_max": {
                    name: param.data.max().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "parameters_min": {
                    name: param.data.min().item() for name, param in net.named_parameters() if param.requires_grad
                },
                "random_parameter": dict(net.named_parameters())[key_to_track][0][0],
            })
            
            if batch_index >= max_batches: 
                break
                    
        end_time = time.time() 
        print(f'Epoch {epoch} training time: {end_time - start_time}')
           
        start_test_time = time.time() 
        _, accuracy, _ = vqa_test(net, test_dataloader)
        end_test_time = time.time() 
        print(f'Epoch {epoch} test time: {end_test_time - start_test_time}')
        
        losses_epoch[epoch] = accumulated_loss / sample_count
        accuracies[epoch] = accuracy
        times_accuracy[end_time - start_time] = accuracy
        times_loss[end_time - start_time] = accumulated_loss / sample_count
        
        print(f'Epoch {epoch} accuracy: {accuracy}')
    
        if epoch % checkpoint_interval == 0:
            with open(results_dir + 'train_losses_epoch.json', 'w+') as f:
                json.dump(losses_epoch, f)
            with open(results_dir + 'train_losses_iter.json', 'w+') as f:
                json.dump(losses_iter, f)
            with open(results_dir + 'accuracies.json', 'w+') as f:
                json.dump(accuracies, f)
            with open(results_dir + 'times_accuracy.json', 'w+') as f:
                json.dump(times_accuracy, f)
            with open(results_dir + 'times_loss.json', 'w+') as f:
                json.dump(times_loss, f)
            
        print('')

def vqa_test(net, test_dataloader):
    average_accuracy = 0
    accumulated_loss = 0
    sample_count = 0
    net.eval()
    
    accuracies = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            batch_dict = {
                'input_ids': batch['input_ids'].to(DEVICE),
                'attention_mask': batch['attention_mask'].to(DEVICE),
                'pixel_values': batch['pixel_values'].to(DEVICE),
                'image_grid_thw': batch['image_grid_thw'].to(DEVICE),
                'labels': batch['labels'].to(DEVICE),
            }
            
            try:
                prediction_tokens = net.generate(
                    **batch_dict, 
                    max_new_tokens=150, #15,
                    # num_beams=4,
                    # do_sample=True,
                    temperature=0.7,
                    top_p = 0.95, #0,
                    # top_k = 1,
                    # repetition_penalty=1,
                )
                predictions = processor.batch_decode(prediction_tokens, skip_special_tokens=True)
                extracted_answers = extract_answer_from_array(predictions)
                
                for extracted_answer, ground_truth in zip(extracted_answers, batch['answers']):
                    accuracies.append(
                        min(sum(1 for item in ground_truth.split(';') 
                                if (item.strip().lower() in extracted_answer.lower()) 
                                or (extracted_answer.lower() in item.strip().lower()))/3, 1
                            )
                        )
            except Exception as e:
                print(e)
            
            sample_count += batch_dict['input_ids'].shape[0] 
            
    accumulated_loss /= sample_count
    average_accuracy = float(sum(accuracies))/len(accuracies)
    print('Test average accuracy: ', average_accuracy)
    return accumulated_loss, average_accuracy, sample_count

 
def get_quantized_model_path(model_name, dataset=None):
    base_path = "./models/models--4bit-gptq-quantized-"
    if "Llama-3.1" in model_name:
        if dataset in ["agnews"]:
            return f"{base_path}llama3-1-8B-classification-4-class"
        elif dataset in ["boolq", "multirc"]:
            return f"{base_path}llama3-1-8B-classification-2-class"
        elif "-Instruct" in model_name:
            return f"{base_path}llama3-1-8B-instruct"
        return f"{base_path}llama3-1-8B"
    elif "Mistral" in model_name:
        if dataset in ["agnews"]:
            return f"{base_path}mistral-03-7B-classification-4-class"
        elif dataset in ["boolq", "multirc"]:
            return f"{base_path}mistral-03-7B-classification-2-class"
        elif "-Instruct" in model_name:
            return f"{base_path}mistral-03-7B-instruct"
        return f"{base_path}mistral-03-7B"
    elif "opt" in model_name:
        size = model_name.split("-")[-1]
        if dataset in ["agnews"]:
            return f"{base_path}opt-{size}-classification-4-class"
        elif dataset in ["boolq", "multirc"]:
            return f"{base_path}opt-{size}-classification-2-class"
        return f"{base_path}opt-{size}"
    return None

def load_or_quantize_model(model_name, dataset, num_labels=None, task_type=TaskType.SEQ_CLS):
    quantized_model_path = get_quantized_model_path(model_name, dataset)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer, use_exllama=False)
    
    model_class = AutoModelForSequenceClassification if task_type == TaskType.SEQ_CLS else AutoModelForCausalLM
    
    if quantized_model_path is not None: 
        if not os.path.exists(quantized_model_path):
            print(f'Quantizing the model: {model_name}')
            if task_type == TaskType.SEQ_CLS:
                net = model_class.from_pretrained(model_name, device_map="cuda", quantization_config=gptq_config, cache_dir="./models", num_labels=num_labels)
            else:
                net = model_class.from_pretrained(model_name, device_map="cuda", quantization_config=gptq_config, cache_dir="./models")
            net.save_pretrained(quantized_model_path)
            tokenizer.save_pretrained(quantized_model_path)
            print(f'Quantized model saved at {quantized_model_path}')
            del net
        
        print(f'Loading quantized model from {quantized_model_path}')
        
        if task_type == TaskType.SEQ_CLS:
            if '1080' in GPU_NAME or '2080' in GPU_NAME or 'M40' in GPU_NAME or 'RTX 8000' in GPU_NAME or 'TITAN X' in GPU_NAME:
                net = model_class.from_pretrained(quantized_model_path, device_map="auto", quantization_config=gptq_config, cache_dir="./models", attn_implementation="eager", torch_dtype=torch.float32, num_labels=num_labels)
            else:
                net = model_class.from_pretrained(quantized_model_path, device_map="auto", quantization_config=gptq_config, cache_dir="./models", attn_implementation="eager", torch_dtype=torch.bfloat16, num_labels=num_labels)
        else:
            if '1080' in GPU_NAME or '2080' in GPU_NAME or 'M40' in GPU_NAME or 'RTX 8000' in GPU_NAME or 'TITAN X' in GPU_NAME:
                net = model_class.from_pretrained(quantized_model_path, device_map="auto", quantization_config=gptq_config, cache_dir="./models", attn_implementation="eager", torch_dtype=torch.float32)
            else:
                net = model_class.from_pretrained(quantized_model_path, device_map="auto", quantization_config=gptq_config, cache_dir="./models", attn_implementation="eager", torch_dtype=torch.bfloat16)
    else:
        print(f'Loading model from {model_name}')
        if task_type == TaskType.SEQ_CLS:
            net = model_class.from_pretrained(model_name, device_map="cuda", cache_dir="./models", attn_implementation="eager", num_labels=num_labels)
        else:
            net = model_class.from_pretrained(model_name, device_map="cuda", cache_dir="./models", attn_implementation="eager",)
    
    
    target_modules = None
    key_to_track = None
    if 'Llama' in model_name:
        target_modules = ['q_proj', 'v_proj',]
        key_to_track = 'base_model.model.model.layers.20.self_attn.v_proj.lora_B.default.weight'
    elif 'opt' in model_name:
        key_to_track = 'base_model.model.model.decoder.layers.20.self_attn.v_proj.lora_B.default.weight'
    elif 'roberta' in model_name:
        key_to_track = 'base_model.model.roberta.encoder.layer.8.attention.self.value.lora_B.default.weight'
    elif 'bert' in model_name:
        key_to_track = 'base_model.model.bert.encoder.layer.8.attention.self.value.lora_B.default.weight'
        
    if task_type == TaskType.SEQ_CLS:
        net.config.pad_token_id = net.config.eos_token_id
        
    return net, tokenizer, task_type, target_modules, key_to_track

def load_qwen2_vl():
    model_name = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4"
    net = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", cache_dir="./models", attn_implementation="eager")
    processor = AutoProcessor.from_pretrained(model_name, cache_dir="./models")
    
    target_modules = set()
    multimodal_keywords = ['mlp', 'lm_head', 'visual']
    
    for name, module in net.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if 'Linear' in str(type(module)):
            target_modules.add(name.split('.', 1)[1])
        if 'lm_head' in target_modules: # needed for 16-bit
            target_modules.remove('lm_head')
    target_modules = list(target_modules) 
    
    return net, processor, TaskType.CAUSAL_LM, target_modules, 'base_model.model.model.layers.20.self_attn.q_proj.lora_A.default.weight'

 
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition_count',)
    parser.add_argument('--epochs',)
    parser.add_argument('--batch_size',)
    parser.add_argument('--lr',)
    parser.add_argument('--optimizer',)
    parser.add_argument('--momentum',)
    parser.add_argument('--max_batches',)
    parser.add_argument('--model_name',)
    parser.add_argument('--max_seq_len',)
    parser.add_argument('--peft_method',)
    parser.add_argument('--lora_r',)
    parser.add_argument('--lora_alpha',)
    parser.add_argument('--dataset',)
    parser.add_argument('--dirichlet_distribution',)
    parser.add_argument('--partition_id',)
    parser.add_argument('--perturbation_count',)
    parser.add_argument('--perturbation_var',)
    parser.add_argument('--perturbation_mean',)
    parser.add_argument('--prob_to_keep',)
    parser.add_argument('--accumulation_steps',)
    parser.add_argument('--mask_beta',)
    parser.add_argument('--jvp_range',)
    parser.add_argument('--grad_value_range',)
    parser.add_argument('--grad_norm_range',)
    parser.add_argument('--fixed_seed',)
    parser.add_argument('--random_seed',)
    parser.add_argument('--checkpoint_interval',)
    parser.add_argument('--calibrated',)
    
    parser.add_argument('--experiment_name',)
    parser.add_argument('--results_dir',)
    
    args = parser.parse_args()
    hyperparameters = {}

    partition_count = int(args.partition_count)
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    optimizer_name = args.optimizer
    momentum = float(args.momentum)
    max_batches = int(args.max_batches)
    model_name = args.model_name
    max_seq_len = int(args.max_seq_len)
    peft_method = args.peft_method
    lora_r = int(args.lora_r)
    lora_alpha = float(args.lora_alpha)
    dataset = args.dataset
    dirichlet_distribution = float(args.dirichlet_distribution)
    partition_id = args.partition_id
    perturbation_count = int(args.perturbation_count)
    perturbation_var = float(args.perturbation_var)
    perturbation_mean = float(args.perturbation_mean)
    prob_to_keep = float(args.prob_to_keep)
    accumulation_steps = int(args.accumulation_steps)
    mask_beta = float(args.mask_beta)
    jvp_range = float(args.jvp_range)
    grad_value_range = float(args.grad_value_range)
    grad_norm_range = float(args.grad_norm_range)
    fixed_seed = args.fixed_seed
    random_seed = int(args.random_seed)
    checkpoint_interval = int(args.checkpoint_interval)
    calibrated = args.calibrated
    experiment_name = args.experiment_name
    results_dir = args.results_dir
    
    hyperparameters['partition_count'] = partition_count
    hyperparameters['epochs'] = epochs
    hyperparameters['batch_size'] = batch_size
    hyperparameters['lr'] = lr
    hyperparameters['optimizer'] = optimizer_name
    hyperparameters['momentum'] = momentum
    hyperparameters['max_batches'] = max_batches
    hyperparameters['model_name'] = model_name
    hyperparameters['max_seq_len'] = max_seq_len
    hyperparameters['peft_method'] = peft_method
    hyperparameters['lora_r'] = lora_r
    hyperparameters['lora_alpha'] = lora_alpha
    hyperparameters['dataset'] = dataset
    hyperparameters['dirichlet_distribution'] = dirichlet_distribution
    hyperparameters['partition_id'] = partition_id
    hyperparameters['dataset'] = dataset
    hyperparameters['perturbation_count'] = perturbation_count
    hyperparameters['perturbation_var'] = perturbation_var
    hyperparameters['perturbation_mean'] = perturbation_mean
    hyperparameters['prob_to_keep'] = prob_to_keep
    hyperparameters['accumulation_steps'] = accumulation_steps
    hyperparameters['mask_beta'] = mask_beta
    hyperparameters['jvp_range'] = jvp_range
    hyperparameters['grad_value_range'] = grad_value_range
    hyperparameters['grad_norm_range'] = grad_norm_range
    hyperparameters['fixed_seed'] = True if fixed_seed == "True" else False
    hyperparameters['random_seed'] = random_seed
    hyperparameters['checkpoint_interval'] = checkpoint_interval
    hyperparameters['experiment_name'] = experiment_name
    hyperparameters['results_dir'] = results_dir
    hyperparameters['calibrated'] = True if calibrated == "True" else False
    calibrated = hyperparameters['calibrated'] 
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    with open(results_dir + 'hyperparameters.json', 'w+') as f:
        json.dump(hyperparameters, f)
        
    random.seed(random_seed)    
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    
    print('Starting ' + experiment_name)
    print('Device: ' + DEVICE)
    
    target_modules = None
    task_type = None
    key_to_track = None
    
    ## Load Models 
    if "Qwen2-VL" in model_name:
        net, processor, task_type, target_modules, key_to_track = load_qwen2_vl()
    elif model_name in ["roberta", "bert"]:
        num_labels = 4 if dataset == "agnews" else 2
        net, tokenizer, task_type, target_modules, key_to_track = load_or_quantize_model(model_name, dataset, num_labels=num_labels)
    else:
        num_labels = 4 if dataset == "agnews" else (2 if dataset in ["boolq", "multirc"] else None)
        task_type = TaskType.SEQ_CLS if num_labels else TaskType.CAUSAL_LM
        net, tokenizer, task_type, target_modules, key_to_track = load_or_quantize_model(model_name, dataset, num_labels=num_labels, task_type=task_type)
    
    # PEFT
    if peft_method == 'lora':
        if target_modules == None:
            lora_config = LoraConfig(
                task_type=task_type, 
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.1,
                inference_mode=False
            )
        else:
            lora_config = LoraConfig(
                task_type=task_type, 
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.1,
                inference_mode=False,
                target_modules = target_modules
            )
        net = get_peft_model(net, lora_config)
        # net = get_peft_model(net.model, lora_config)
    
    param_count = sum([np.prod(p.size()) for p in net.parameters()])
    trainable_param_count = sum([np.prod(p.size()) for p in net.parameters() if p.requires_grad])
    print('Param count (not counting quantized layers): ', param_count, flush=True)
    print('Trainble param count: ', trainable_param_count, flush=True)
    
    wandb.init(
        project='MoreForwardModeADBeta'+dataset,
        name=experiment_name,
        config=hyperparameters
    )
    
    ## Load Datasets and Call Trainers
    if dataset == 'agnews':
        train_dataloader_dict, test_dataloader_dict = agnews_sequence_classification_dataloader.get_federated_datasets(
            dirichlet_parameter=dirichlet_distribution,
            num_clients=partition_count, 
            train_client_batch_size=batch_size,
            tokenizer_name=model_name
        )
        train_dataloader, test_dataloader = train_dataloader_dict[partition_id], test_dataloader_dict[partition_id]
        print(net)
        print('Initial test accuracy', str(text_classification_test(net, test_dataloader)[1]))
        if calibrated == False:
            gradient_accumulation_text_classification_train(net, train_dataloader, test_dataloader)
        else:
            calibrated_gradients_text_classification_train(net, train_dataloader, test_dataloader)
    elif dataset == 'boolq':
        train_dataloader, test_dataloader = boolq_dataloader.get_centralized_dataset(
            train_batch_size=batch_size,
            tokenizer_name=model_name,
        )
        print(net)
        print('Initial test accuracy', str(text_classification_test(net, test_dataloader)[1]))
        if calibrated == False:
            gradient_accumulation_text_classification_train(net, train_dataloader, test_dataloader)
        else:
            calibrated_gradients_text_classification_train(net, train_dataloader, test_dataloader)
    elif dataset == 'gsm8k':
        train_dataloader, test_dataloader = gsm8k_dataloader.get_centralized_dataset(
            train_batch_size=batch_size,
            tokenizer_name=model_name,
        )
        # print(net)
        print('Initial test accuracy', str(text_generation_gsm8k_test(net, test_dataloader)[1]))
        if calibrated == False:
            gradient_accumulation_text_generation_train(net, train_dataloader, test_dataloader)
        else:
            calibrated_gradients_text_generation_train(net, train_dataloader, test_dataloader)
    elif dataset == 'mmlu':
        train_dataloader, test_dataloader = mmlu_dataloader.get_centralized_dataset(
            train_batch_size=batch_size,
            tokenizer_name=model_name,
        )
        print('Initial test accuracy', str(text_generation_mmlu_test(net, test_dataloader)[1]))
        if calibrated == False:
            gradient_accumulation_text_generation_train(net, train_dataloader, test_dataloader)
        else:
            calibrated_gradients_text_generation_train(net, train_dataloader, test_dataloader)
    elif dataset == 'multirc':
        train_dataloader, test_dataloader = multirc_dataloader.get_centralized_dataset(
            train_batch_size=batch_size,
            tokenizer_name=model_name,
        )
        # print(net)
        print('Initial test accuracy', str(text_classification_test(net, test_dataloader)[1]))
        if calibrated == False:
            gradient_accumulation_text_classification_train(net, train_dataloader, test_dataloader)
        else:
            calibrated_gradients_text_classification_train(net, train_dataloader, test_dataloader)
    elif dataset == 'vqav2':
        train_dataloader, test_dataloader = vqav2_dataloader.get_centralized_dataset(
            processor,
            train_batch_size=batch_size,
        )
        print(net)
        print('Initial test accuracy', str(vqa_test(net, test_dataloader,)[1]))
        if calibrated == False:
            gradient_accumulation_vqa_train(net, train_dataloader, test_dataloader,)
        else:
            calibrated_gradients_vqa_train(net, train_dataloader, test_dataloader,)
    elif dataset == 'gqa':
        train_dataloader, test_dataloader = gqa_dataloader.get_centralized_dataset(
            processor,
            train_batch_size=batch_size,
        )
        print(net)
        print('Initial test accuracy', str(vqa_test(net, test_dataloader,)[1]))
        if calibrated == False:
            gradient_accumulation_vqa_train(net, train_dataloader, test_dataloader,)
        else:
            calibrated_gradients_vqa_train(net, train_dataloader, test_dataloader,)
    elif dataset == 'textvqa':
        train_dataloader, test_dataloader = textvqa_dataloader.get_centralized_dataset(
            processor,
            train_batch_size=batch_size,
        )
        print(net)
        print('Initial test accuracy', str(vqa_test(net, test_dataloader,)[1]))
        if calibrated == False:
            gradient_accumulation_vqa_train(net, train_dataloader, test_dataloader,)
        else:
            calibrated_gradients_vqa_train(net, train_dataloader, test_dataloader,)
    else:
        print('Trainer not defined. Exiting.')

