import torch

def get_orthogonal_tensor(tensor: torch.Tensor) -> torch.Tensor:
    flat_tensor = tensor.reshape(-1)
    n = flat_tensor.numel()
    
    if n < 2:
        raise ValueError("Tensor must have at least 2 elements")
    
    perp = torch.zeros_like(flat_tensor)
    
    for i in range(0, n-1, 2):
        if flat_tensor[i] != 0 or flat_tensor[i+1] != 0:
            perp[i] = -flat_tensor[i+1]
            perp[i+1] = flat_tensor[i]
            break
    
    flat_tensor = flat_tensor / torch.norm(flat_tensor)
    perp = perp / torch.norm(perp)
    
    return perp.reshape(tensor.shape)

## An old attempt at generating orthogonal vectors    
#  def get_orthogonal_tensor(tensor: torch.Tensor) -> torch.Tensor:   
    # V_norm = V / V.norm(dim=1, keepdim=True)

    # if torch.isnan(V_norm).any():
    #     print('Nans in V_norm')
    # # Generate a random matrix R of the same shape
    # R = torch.randn_like(V)
    # if torch.isnan(R).any():
    #     print('Nans in R')

    # # Make R orthogonal to V row-wise (90° vectors)
    # dot_product = torch.sum(V_norm * R, dim=1, keepdim=True)  # Row-wise dot product
    # print('dot_product', dot_product)
    # U_90 = R - dot_product * V_norm  # Project R onto V and subtract
    # print(U_90)
    # if torch.isnan(U_90).any():
    #     print('Nans in U_90 1')
    # # U_90 = U_90 / U_90.norm(dim=1, keepdim=True) * V.norm(dim=1, keepdim=True)  # Normalize to match original norms
    # if torch.isnan(U_90).any():
    #     print('Nans in U_90 2')

    # return U_90
    
def invert_wide_tensor(x):
    x = x.reshape(1, -1)
    x_inv = torch.pinverse(x)
    x_normalized = x.T / (torch.norm(x) ** 2)
    eps = 1e-10  # Small constant for numerical stability
    x_stable = x.T / (torch.norm(x) ** 2 + eps)
    
    return x_stable 

def sample_like(tensor, distribution, variance=1.0):
    return distribution.sample(tensor.shape) * torch.sqrt(torch.tensor(variance))

def normalize_tensor(x, a=-1, b=1):
    min_val = x.min()
    max_val = x.max()
    return (b - a) * (x - min_val) / (max_val - min_val) + a

def top_k_percent_mask(tensor, k):
    abs_tensor = tensor.abs().flatten()
    k = max(1, int((k / 100) * abs_tensor.numel()))
    
    if k == abs_tensor.numel():
        threshold = abs_tensor.min()
    else:
        threshold = abs_tensor.topk(k, sorted=False).values.min()
    
    mask = tensor.abs() >= threshold
    return mask

def top_k_percent_mask_v2(tensor, k_percent):
    k_count = tensor.numel() * k_percent // 100
    _, topk_indices = torch.topk(torch.abs(tensor), k_count)
    mask = torch.zeros_like(tensor, dtype=torch.bool)
    mask[topk_indices] = True
    return mask

def apply_gradient_mask(param, mask):
    param.grad.data *= mask