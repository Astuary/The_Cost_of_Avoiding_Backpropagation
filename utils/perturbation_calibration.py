import math
import random

import torch
import torch.autograd.forward_ad as fwAD
from torch.autograd.profiler import record_function

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_subset_orthogonal_tangents_for_calibration_phase(param_names, param_sizes):
    tangents_0_degree = {
        name: (
            torch.randn(size) * torch.sqrt(torch.tensor(perturbation_var))
        ).to(DEVICE) 
        for size, name in zip(param_sizes, param_names)
    }
    tangents_90_degree = {
        name: (
            get_orthogonal_tensor(tangent)
        ).to(DEVICE) 
        for name, tangent in tangents_0_degree.items()
    }
    tangents_180_degree = {
        name: (
            tangent * -1
        ).to(DEVICE) 
        for name, tangent in tangents_0_degree.items()
    }
    tangents_270_degree = {
        name: (
            tangent * -1
        ).to(DEVICE) 
        for name, tangent in tangents_90_degree.items()
    }
    return [tangents_0_degree, tangents_90_degree, tangents_180_degree, tangents_270_degree]

def generate_subset_random_tangents_for_calibration_phase(param_names, param_sizes, perturbation_var, tangent_count=4):
    tangent_list = []
    for i in range(tangent_count):
        tangents = {
            name: (
                torch.randn(size) * torch.sqrt(torch.tensor(perturbation_var))
            ).to(DEVICE) 
            for size, name in zip(param_sizes, param_names)
        }
        tangent_list.append(tangents)
    return tangent_list

def generate_tangents_for_calibration_phase(trainable_layer_names, params, batch_dict):
    tangents = {}
    for layer_id in range(0, len(trainable_layer_names) - 1, 2):
        name_1 = trainable_layer_names[layer_id]
        name_2 = trainable_layer_names[layer_id+1]
        list_of_tangents = generate_subset_random_tangents_for_calibration_phase(
            [name_1, name_2],
            [params[name_1].shape, params[name_2].shape],
        )
    
        best_tangent_id = None
        best_jvp = -math.inf
        for tangent_id, subset_tangents in enumerate(list_of_tangents):
            with fwAD.dual_level(), torch.no_grad():
                new_state_dict = {}
                for n, v in subset_tangents.items():
                    new_state_dict[n] = fwAD.make_dual(params[n], v)
                net.load_state_dict(new_state_dict, strict=False)
                del new_state_dict
                
                outputs = net(**batch_dict)                
                jvp = fwAD.unpack_dual(outputs.loss).tangent
                if jvp_range > 0.0:
                    jvp = torch.clamp(jvp, min=-jvp_range, max=jvp_range)
                # print('Calibration jvp: ', jvp, flush=True)
            
            if torch.isnan(jvp):
                print('Skipping NaN jvp during calibration', flush=True)
                # break
            else:
                if jvp > best_jvp:
                    best_jvp = jvp
                    best_tangent_id = tangent_id
                    
        if best_tangent_id is not None:
            for name, v in list_of_tangents[best_tangent_id].items():
                tangents[name] = v
                
    if len(trainable_layer_names) % 2 != 0:
        name = trainable_layer_names[-1]
        list_of_tangents = generate_subset_random_tangents_for_calibration_phase(
            [name],
            [params[name].shape],
        )
    
        best_tangent_id = None
        best_jvp = -math.inf
        for tangent_id, subset_tangents in enumerate(list_of_tangents):
            
            with fwAD.dual_level(), torch.no_grad(), torch.cuda.amp.autocast():
                new_state_dict = {}
                for n, v in subset_tangents.items():
                    new_state_dict[n] = fwAD.make_dual(params[n], v)
                net.load_state_dict(new_state_dict, strict=False)
                del new_state_dict
                
                outputs = net(**batch_dict)                
                jvp = fwAD.unpack_dual(outputs.loss).tangent
                if jvp_range > 0.0:
                    jvp = torch.clamp(jvp, min=-jvp_range, max=jvp_range)
                # print('Calibration jvp: ', jvp, flush=True)
            
            if torch.isnan(jvp):
                print('Skipping NaN jvp during calibration', flush=True)
            else:
                if jvp > best_jvp:
                    best_jvp = jvp
                    best_tangent_id = tangent_id
                    
        if best_tangent_id is not None:
            for name, v in list_of_tangents[best_tangent_id].items():
                tangents[name] = v
                
    return tangents

def generate_gradients_for_calibration_phase(
    net, trainable_layer_names, params, batch_dict, perturbation_var, jvp_range
    ):
    tangents = {}
    gradients = {}
    accumulated_loss = torch.tensor(0.0)
    accumulated_jvp = torch.tensor(0.0)
    
    for layer_id in range(0, len(trainable_layer_names) - 1, 2):
        name_1 = trainable_layer_names[layer_id]
        name_2 = trainable_layer_names[layer_id+1]
        list_of_tangents = generate_subset_random_tangents_for_calibration_phase(
            [name_1, name_2],
            [params[name_1].shape, params[name_2].shape],
            perturbation_var
        )
    
        best_tangent_id = None
        best_jvp = math.inf
        for tangent_id, subset_tangents in enumerate(list_of_tangents):
            with fwAD.dual_level(), torch.no_grad():
                new_state_dict = {}
                for n, v in subset_tangents.items():
                    new_state_dict[n] = fwAD.make_dual(params[n], v)
                net.load_state_dict(new_state_dict, strict=False)
                del new_state_dict
                
                outputs = net(**batch_dict)                
                jvp = fwAD.unpack_dual(outputs.loss).tangent
                if jvp_range > 0.0:
                    jvp = torch.clamp(jvp, min=-jvp_range, max=jvp_range)
            
            if torch.isnan(jvp):
                print('Skipping NaN jvp during calibration', flush=True)
            else:
                if abs(jvp) < abs(best_jvp):
                    best_jvp = jvp
                    best_tangent_id = tangent_id
                    loss = outputs.loss
        accumulated_loss += loss.to('cpu')
        accumulated_jvp += best_jvp.to('cpu')
                    
        if best_tangent_id is not None:
            for name, v in list_of_tangents[best_tangent_id].items():
                gradients[name] = v * best_jvp
                tangents[name] = v
                
    if len(trainable_layer_names) % 2 != 0:
        name = trainable_layer_names[-1]
        list_of_tangents = generate_subset_random_tangents_for_calibration_phase(
            [name],
            [params[name].shape],
            perturbation_var
        )
    
        best_tangent_id = None
        best_jvp = math.inf
        for tangent_id, subset_tangents in enumerate(list_of_tangents):
            
            with fwAD.dual_level(), torch.no_grad(), torch.cuda.amp.autocast():
                new_state_dict = {}
                for n, v in subset_tangents.items():
                    new_state_dict[n] = fwAD.make_dual(params[n], v)
                net.load_state_dict(new_state_dict, strict=False)
                del new_state_dict
                
                outputs = net(**batch_dict)                
                jvp = fwAD.unpack_dual(outputs.loss).tangent
                if jvp_range > 0.0:
                    jvp = torch.clamp(jvp, min=-jvp_range, max=jvp_range)
                # print('Calibration jvp: ', jvp, flush=True)
            
            if torch.isnan(jvp):
                print('Skipping NaN jvp during calibration', flush=True)
            else:
                if abs(jvp) < abs(best_jvp):
                    best_jvp = jvp
                    best_tangent_id = tangent_id
                    loss = outputs.loss
        accumulated_loss += loss.to('cpu')
        accumulated_jvp += best_jvp.to('cpu')
                    
        if best_tangent_id is not None:
            for name, v in list_of_tangents[best_tangent_id].items():
                gradients[name] = v * best_jvp
                tangents[name] = v
                
    return (gradients, tangents, 
        accumulated_loss / len(trainable_layer_names), 
        accumulated_jvp / len(trainable_layer_names)
    )

def generate_gradients_for_zero_order_calibration_phase(
    net, trainable_layer_names, params, 
    batch_dict, perturbation_step_size, 
    perturbation_var, projected_gradient_range
    ):
    print('Calibrating.', flush=True)
    tangents = {}
    gradients = {}
    accumulated_loss = torch.tensor(0.0)
    accumulated_projected_gradient = torch.tensor(0.0)
    
    for layer_id in range(0, len(trainable_layer_names) - 1, 2):
        name_1 = trainable_layer_names[layer_id]
        name_2 = trainable_layer_names[layer_id+1]
        list_of_tangents = generate_subset_random_tangents_for_calibration_phase(
            [name_1, name_2],
            [params[name_1].shape, params[name_2].shape],
            perturbation_var
        )
    
        best_tangent_id = None
        best_projected_gradient = math.inf
        for tangent_id, subset_tangents in enumerate(list_of_tangents):
            perturbation_seed = random.randint(0, 2**31 - 1)
            
            torch.manual_seed(perturbation_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(perturbation_seed)
                torch.cuda.manual_seed_all(perturbation_seed)
                
            ## Normal distribution
            positively_perturbed_params = {
                name: p + (subset_tangents[name]).to(DEVICE) 
                for name, p in net.named_parameters() if name in subset_tangents.keys()
            }
            net.load_state_dict(positively_perturbed_params, strict=False)
            del positively_perturbed_params
                
            with torch.no_grad(), record_function("## forward 1 ##"): # with torch.cuda.amp.autocast():
                outputs = net(**batch_dict)
                first_loss = outputs.loss
                
            torch.manual_seed(perturbation_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(perturbation_seed)
                torch.cuda.manual_seed_all(perturbation_seed)
                
            negatively_perturbed_params = {
                name: p - (2 * subset_tangents[name]).to(DEVICE) 
                for name, p in net.named_parameters() if name in subset_tangents.keys()
            }
            net.load_state_dict(negatively_perturbed_params, strict=False)
            del negatively_perturbed_params    
                
            with torch.no_grad(), record_function("## forward 2 ##"): # with torch.cuda.amp.autocast():
                outputs = net(**batch_dict)
                second_loss = outputs.loss    
            
            torch.manual_seed(perturbation_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(perturbation_seed)
                torch.cuda.manual_seed_all(perturbation_seed)
                    
            reset_perturbed_params = {
                name: p + (subset_tangents[name]).to(DEVICE) 
                for name, p in net.named_parameters() if name in subset_tangents.keys()
            }
            net.load_state_dict(reset_perturbed_params, strict=False)
            del reset_perturbed_params
            
            loss = (first_loss + second_loss) / 2
            # accumulated_loss += loss.item()
                
            projected_gradient = (first_loss - second_loss) / (2 * perturbation_step_size)
            if projected_gradient_range > 0.0:
                projected_gradient = torch.clamp(projected_gradient, min=-projected_gradient_range, max=projected_gradient_range)
                
            if torch.isnan(projected_gradient):
                print('Skipping NaN projected_gradient', flush=True)
                continue
            else:
                if abs(projected_gradient) < abs(best_projected_gradient):
                    best_projected_gradient = projected_gradient
                    best_tangent_id = tangent_id
        
        accumulated_loss += loss.to('cpu')
        accumulated_projected_gradient += best_projected_gradient.to('cpu')
                    
        if best_tangent_id is not None:
            for name, v in list_of_tangents[best_tangent_id].items():
                gradients[name] = v * best_projected_gradient
                tangents[name] = v
                
    if len(trainable_layer_names) % 2 != 0:
        name = trainable_layer_names[-1]
        list_of_tangents = generate_subset_random_tangents_for_calibration_phase(
            [name],
            [params[name].shape],
            perturbation_var
        )
    
        best_tangent_id = None
        best_projected_gradient = math.inf
        for tangent_id, subset_tangents in enumerate(list_of_tangents):
            perturbation_seed = random.randint(0, 2**31 - 1)
            
            torch.manual_seed(perturbation_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(perturbation_seed)
                torch.cuda.manual_seed_all(perturbation_seed)
                
            ## Normal distribution
            positively_perturbed_params = {
                name: p + (subset_tangents[name]).to(DEVICE) 
                for name, p in net.named_parameters() if name in subset_tangents.keys()
            }
            net.load_state_dict(positively_perturbed_params, strict=False)
            del positively_perturbed_params
                
            with torch.no_grad(), record_function("## forward 1 ##"): # with torch.cuda.amp.autocast():
                outputs = net(**batch_dict)
                first_loss = outputs.loss
                
            torch.manual_seed(perturbation_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(perturbation_seed)
                torch.cuda.manual_seed_all(perturbation_seed)
                
            negatively_perturbed_params = {
                name: p - (2 * subset_tangents[name]).to(DEVICE) 
                for name, p in net.named_parameters() if name in subset_tangents.keys()
            }
            net.load_state_dict(negatively_perturbed_params, strict=False)
            del negatively_perturbed_params    
                
            with torch.no_grad(), record_function("## forward 2 ##"): # with torch.cuda.amp.autocast():
                outputs = net(**batch_dict,)
                second_loss = outputs.loss    
            
            torch.manual_seed(perturbation_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(perturbation_seed)
                torch.cuda.manual_seed_all(perturbation_seed)
                    
            reset_perturbed_params = {
                name: p + (subset_tangents[name]).to(DEVICE) 
                for name, p in net.named_parameters() if name in subset_tangents.keys()
            }
            net.load_state_dict(reset_perturbed_params, strict=False)
            del reset_perturbed_params
            
            loss = (first_loss + second_loss) / 2
                
            projected_gradient = (first_loss - second_loss) / (2 * perturbation_step_size)
            if projected_gradient_range > 0.0:
                projected_gradient = torch.clamp(projected_gradient, min=-projected_gradient_range, max=projected_gradient_range)
                
            if torch.isnan(projected_gradient):
                print('Skipping NaN projected_gradient', flush=True)
                continue
            else:
                if abs(projected_gradient) < abs(best_projected_gradient):
                    best_projected_gradient = projected_gradient
                    best_tangent_id = tangent_id
        
        accumulated_loss += loss.to('cpu')
        accumulated_projected_gradient += best_projected_gradient.to('cpu')
                    
        if best_tangent_id is not None:
            for name, v in list_of_tangents[best_tangent_id].items():
                gradients[name] = v * best_projected_gradient
                tangents[name] = v
                
    return (gradients, tangents, 
        accumulated_loss / len(trainable_layer_names), 
        accumulated_projected_gradient / len(trainable_layer_names)
    )

def generate_gradients_jvp_for_a_range_for_calibration_phase(trainable_layer_names, params, batch_dict):
    jvp_high = 1
    jvp_low = -1
    
    tangents = {}
    gradients = {}
    accumulated_loss = torch.tensor(0.0)
    accumulated_jvp = torch.tensor(0.0)
    
    for layer_id in range(0, len(trainable_layer_names) - 1, 2):
        name_1 = trainable_layer_names[layer_id]
        name_2 = trainable_layer_names[layer_id+1]
        list_of_tangents = generate_subset_random_tangents_for_calibration_phase(
            [name_1, name_2],
            [params[name_1].shape, params[name_2].shape],
        )
    
        best_tangent_id = None
        best_jvp = None
        for tangent_id, subset_tangents in enumerate(list_of_tangents):
            with fwAD.dual_level(), torch.no_grad():
                new_state_dict = {}
                for n, v in subset_tangents.items():
                    new_state_dict[n] = fwAD.make_dual(params[n], v)
                net.load_state_dict(new_state_dict, strict=False)
                del new_state_dict
                
                outputs = net(**batch_dict)                
                jvp = fwAD.unpack_dual(outputs.loss).tangent
                if jvp_range > 0.0:
                    jvp = torch.clamp(jvp, min=-jvp_range, max=jvp_range)
            
            if torch.isnan(jvp):
                print('Skipping NaN jvp during calibration', flush=True)
            else:
                if (jvp < jvp_high) and (jvp > jvp_low):
                    best_jvp = jvp
                    best_tangent_id = tangent_id
                    loss = outputs.loss
        if best_jvp is None:
            best_jvp = jvp
            best_tangent_id = tangent_id
            loss = outputs.loss
        
        print('During calibration, best jvp:', best_jvp)    
        accumulated_loss += loss.to('cpu')
        accumulated_jvp += best_jvp.to('cpu')
                    
        if best_tangent_id is not None:
            for name, v in list_of_tangents[best_tangent_id].items():
                gradients[name] = v * best_jvp
                tangents[name] = v
                
    if len(trainable_layer_names) % 2 != 0:
        name = trainable_layer_names[-1]
        list_of_tangents = generate_subset_random_tangents_for_calibration_phase(
            [name],
            [params[name].shape],
        )
    
        best_tangent_id = None
        best_jvp = None
        for tangent_id, subset_tangents in enumerate(list_of_tangents):
            
            with fwAD.dual_level(), torch.no_grad(), torch.cuda.amp.autocast():
                new_state_dict = {}
                for n, v in subset_tangents.items():
                    new_state_dict[n] = fwAD.make_dual(params[n], v)
                net.load_state_dict(new_state_dict, strict=False)
                del new_state_dict
                
                outputs = net(**batch_dict)                
                jvp = fwAD.unpack_dual(outputs.loss).tangent
                if jvp_range > 0.0:
                    jvp = torch.clamp(jvp, min=-jvp_range, max=jvp_range)
                # print('Calibration jvp: ', jvp, flush=True)
            
            if torch.isnan(jvp):
                print('Skipping NaN jvp during calibration', flush=True)
            else:
                if (jvp < jvp_high) and (jvp > jvp_low):
                    best_jvp = jvp
                    best_tangent_id = tangent_id
                    loss = outputs.loss
        if best_jvp is None:
            best_jvp = jvp
            best_tangent_id = tangent_id
            loss = outputs.loss
        print('During calibration, best jvp:', best_jvp)    
            
        accumulated_loss += loss.to('cpu')
        accumulated_jvp += best_jvp.to('cpu')
                    
        if best_tangent_id is not None:
            for name, v in list_of_tangents[best_tangent_id].items():
                gradients[name] = v * best_jvp
                tangents[name] = v
                
    return (gradients, tangents, 
        accumulated_loss / len(trainable_layer_names), 
        accumulated_jvp / len(trainable_layer_names)
    )

def generate_dependent_gradients_for_calibration_phase(
    optimizer, net, batch_dict
    ):
    net_snapshot = copy.deepcopy(net)
    
    gradients = {}
    params = {name: p for name, p in net_snapshot.named_parameters() if p.requires_grad}
    
    for layer_id in range(0, len(params.keys()) - 1, 2):
        name_1 = list(params.keys())[layer_id]
        name_2 = list(params.keys())[layer_id + 1]
        list_of_tangents = generate_subset_random_tangents_for_calibration_phase(
            [name_1, name_2],
            [params[name_1].shape,params[name_2].shape],
            perturbation_var
        )
    
        best_tangent_id = None
        best_jvp = math.inf
        for tangent_id, subset_tangents in enumerate(list_of_tangents):
            with fwAD.dual_level(), torch.no_grad():
                new_state_dict = {}
                for n, v in subset_tangents.items():
                    new_state_dict[n] = fwAD.make_dual(params[n], v)
                net_snapshot.load_state_dict(new_state_dict, strict=False)
                del new_state_dict
                
                outputs = net_snapshot(**batch_dict)                
                jvp = fwAD.unpack_dual(outputs.loss).tangent
                if jvp_range > 0.0:
                    jvp = torch.clamp(jvp, min=-jvp_range, max=jvp_range)
            
            if torch.isnan(jvp):
                print('Skipping NaN jvp during calibration', flush=True)
            else:
                if abs(jvp) < abs(best_jvp):
                    best_jvp = jvp
                    best_tangent_id = tangent_id
        
        print('During calibration, best jvp:', best_jvp)            
        if best_tangent_id is not None:
            for name, v in list_of_tangents[best_tangent_id].items():
                gradients[name] = v * best_jvp
                params[name].grad = gradients[name]
        
        optimizer.step()
        optimizer.zero_grad()
                
    if len(params.keys()) % 2 != 0:
        name = list(params.keys())[-1]
        list_of_tangents = generate_subset_random_tangents_for_calibration_phase(
            [name],
            [params[name].shape],
            perturbation_var
        )
    
        best_tangent_id = None
        best_jvp = math.inf
        for tangent_id, subset_tangents in enumerate(list_of_tangents):
            
            with fwAD.dual_level(), torch.no_grad():
                new_state_dict = {}
                for n, v in subset_tangents.items():
                    new_state_dict[n] = fwAD.make_dual(params[n], v)
                net_snapshot.load_state_dict(new_state_dict, strict=False)
                del new_state_dict
                
                outputs = net_snapshot(**batch_dict)                
                jvp = fwAD.unpack_dual(outputs.loss).tangent
                if jvp_range > 0.0:
                    jvp = torch.clamp(jvp, min=-jvp_range, max=jvp_range)
            
            if torch.isnan(jvp):
                print('Skipping NaN jvp during calibration', flush=True)
            else:
                if abs(jvp) < abs(best_jvp):
                    best_jvp = jvp
                    best_tangent_id = tangent_id
                    
        if best_tangent_id is not None:
            for name, v in list_of_tangents[best_tangent_id].items():
                gradients[name] = (v * best_jvp).to(params[name].dtype)
                params[name].grad = gradients[name]
                
        optimizer.step()
        optimizer.zero_grad()
                
    return gradients
