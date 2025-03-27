
import torch
import numpy as np

#----------------------------------------------------------------------------

def get_schedule(num_steps, sigma_min, sigma_max, device=None, schedule_type='polynomial', schedule_rho=7, net=None, diffusion=None):
    """
    Get the time schedule for sampling.

    Args:
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        device: A torch device.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_type: A `float`. Time step exponent.
        net: A pre-trained diffusion model. Required when schedule_type == 'discrete'.
    Returns:
        a PyTorch tensor with shape [num_steps].
    """
    sigma_min = torch.tensor(sigma_min, device=device) if not isinstance(sigma_min, torch.Tensor) else sigma_min
    sigma_max = torch.tensor(sigma_max, device=device) if not isinstance(sigma_max, torch.Tensor) else sigma_max
    if schedule_type == 'polynomial':
        step_indices = torch.arange(num_steps, device=device)
        t_steps = (sigma_max ** (1 / schedule_rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / schedule_rho) - sigma_max ** (1 / schedule_rho))) ** schedule_rho
    elif schedule_type == 'logsnr':
        logsnr_max = -1 * torch.log(torch.tensor(sigma_min))
        logsnr_min = -1 * torch.log(torch.tensor(sigma_max))
        t_steps = torch.linspace(logsnr_min.item(), logsnr_max.item(), steps=num_steps, device=device)
        t_steps = (-t_steps).exp()
    elif schedule_type == 'time_uniform':
        epsilon_s = 1e-3
        # Use PyTorch operations for vp_sigma
        vp_sigma = lambda beta_d, beta_min: lambda t: (torch.exp(torch.tensor(0.5 * beta_d * (t ** 2) + beta_min * t, device=device)) - 1) ** 0.5
        vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * torch.log(sigma ** 2 + 1)).sqrt() - beta_min) / beta_d
        step_indices = torch.arange(num_steps, device=device)
        # Compute vp_beta_d and vp_beta_min using PyTorch
        vp_beta_d = 2 * (torch.log(sigma_min ** 2 + 1) / epsilon_s - torch.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
        vp_beta_min = torch.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d
        t_steps_temp = (1 + step_indices / (num_steps - 1) * (epsilon_s ** (1 / schedule_rho) - 1)) ** schedule_rho
        # Clamp t_steps_temp to avoid large values
        t_steps_temp = torch.clamp(t_steps_temp, min=1e-6, max=1.0)
        t_steps = vp_sigma(vp_beta_d.item(), vp_beta_min.item())(t_steps_temp)
    elif schedule_type == 'discrete':
        # assert net is not None
        # t_steps_min = net.sigma_inv(torch.tensor(sigma_min, device=device))
        # t_steps_max = net.sigma_inv(torch.tensor(sigma_max, device=device))
        # step_indices = torch.arange(num_steps, device=device)
        # t_steps_temp = (t_steps_max + step_indices / (num_steps - 1) * (t_steps_min ** (1 / schedule_rho) - t_steps_max)) ** schedule_rho
        # t_steps = net.sigma(t_steps_temp)

        # Check if the model has sigma and sigma_inv (e.g., for EDM or LDM models)
        has_sigma_inv = net is not None and hasattr(net, 'sigma_inv') and callable(getattr(net, 'sigma_inv')) and hasattr(net, 'sigma') and callable(getattr(net, 'sigma'))
    
        if has_sigma_inv:
            # Original discrete schedule for models with sigma_inv (e.g., EDM, LDM)
            t_steps_min = net.sigma_inv(torch.tensor(sigma_min, device=device))
            t_steps_max = net.sigma_inv(torch.tensor(sigma_max, device=device))
            step_indices = torch.arange(num_steps, device=device)
            t_steps_temp = (t_steps_max + step_indices / (num_steps - 1) * (t_steps_min ** (1 / schedule_rho) - t_steps_max)) ** schedule_rho
            t_steps = net.sigma(t_steps_temp)
        else:
            # Modified discrete schedule for guided diffusion models
            assert diffusion is not None, "Diffusion object must be provided for discrete schedule with guided diffusion"
            assert hasattr(diffusion, 'alphas_cumprod'), "Diffusion object must have alphas_cumprod for discrete schedule"
            alphas_cumprod = diffusion.alphas_cumprod  # Shape: [diffusion_steps]
            print(f"alphas_cumprod shape: {alphas_cumprod.shape}, device: {alphas_cumprod.device if isinstance(alphas_cumprod, torch.Tensor) else 'numpy'}")
            # Convert to torch tensor if it's a numpy array
            if isinstance(alphas_cumprod, np.ndarray):
                alphas_cumprod = torch.from_numpy(alphas_cumprod).to(device,dtype=torch.float32)
            # Check for invalid values in alphas_cumprod
            if (alphas_cumprod < 0).any() or (alphas_cumprod > 1).any():
                raise ValueError(f"alphas_cumprod contains invalid values: min {alphas_cumprod.min().item()}, max {alphas_cumprod.max().item()}")
            sigmas = torch.sqrt(1 - alphas_cumprod)  # Shape: [diffusion_steps]
            # Check for NaN or Inf in sigmas
            if torch.isnan(sigmas).any() or torch.isinf(sigmas).any():
                raise ValueError("sigmas contains NaN or Inf values")
            print(f"sigmas shape: {sigmas.shape}, device: {sigmas.device}, min: {sigmas.min().item()}, max: {sigmas.max().item()}")
    
            # Ensure sigmas, sigma_min, and sigma_max are on the same device
            sigmas = sigmas.to(device)
            sigma_min = sigma_min.to(device)
            sigma_max = sigma_max.to(device)
            # Clamp sigma_min and sigma_max to the range of sigmas
            sigma_min = torch.clamp(sigma_min, sigmas.min(), sigmas.max())
            sigma_max = torch.clamp(sigma_max, sigmas.min(), sigmas.max())
            print(f"After clamping, sigma_min: {sigma_min.item()}, sigma_max: {sigma_max.item()}")
    
            # Find the timesteps corresponding to sigma_min and sigma_max
            timesteps = torch.arange(len(sigmas), device=device, dtype=torch.float32)  # [0, 1, ..., diffusion_steps-1]
            sigma_indices = torch.arange(len(sigmas), device=device, dtype=torch.float32)
            sigma_min_idx = torch.searchsorted(sigmas.flip(0), sigma_min, right=True).float()
            sigma_max_idx = torch.searchsorted(sigmas.flip(0), sigma_max, right=True).float()
            # Convert indices back to timesteps (since we flipped sigmas)
            sigma_min_idx = (len(sigmas) - 1) - sigma_min_idx
            sigma_max_idx = (len(sigmas) - 1) - sigma_max_idx
            print(f"sigma_min_idx: {sigma_min_idx.item()}, sigma_max_idx: {sigma_max_idx.item()}")
    
            # Generate timesteps from sigma_max_idx to sigma_min_idx
            step_indices = torch.arange(num_steps, device=device, dtype=torch.float32)
            t_steps_temp = (sigma_max_idx + step_indices / (num_steps - 1) * (sigma_min_idx - sigma_max_idx)) ** schedule_rho
            print(f"t_steps_temp shape: {t_steps_temp.shape}, device: {t_steps_temp.device}, min: {t_steps_temp.min().item()}, max: {t_steps_temp.max().item()}")
    
            # Map timesteps back to sigmas using interpolation
            # Ensure t_steps_temp is within bounds
            t_steps_temp = torch.clamp(t_steps_temp, 0, len(sigmas) - 1)
            print(f"After clamp, t_steps_temp min: {t_steps_temp.min().item()}, max: {t_steps_temp.max().item()}")
            # Interpolate sigmas at t_steps_temp
            t_steps_temp_int = t_steps_temp.long()
            t_steps_temp_frac = t_steps_temp - t_steps_temp_int.float()
            print(f"t_steps_temp_int shape: {t_steps_temp_int.shape}, device: {t_steps_temp_int.device}, min: {t_steps_temp_int.min().item()}, max: {t_steps_temp_int.max().item()}")
            print(f"t_steps_temp_frac shape: {t_steps_temp_frac.shape}, device: {t_steps_temp_frac.device}, min: {t_steps_temp_frac.min().item()}, max: {t_steps_temp_frac.max().item()}")
            # Compute the next index, ensuring it doesn't exceed bounds
            t_steps_temp_int_plus_1 = torch.clamp(t_steps_temp_int + 1, 0, len(sigmas) - 1)
            # Compute the interpolated values
            sigma_lower = sigmas[t_steps_temp_int]
            sigma_upper = sigmas[t_steps_temp_int_plus_1]
            # Only interpolate where t_steps_temp_int < len(sigmas) - 1
            mask = t_steps_temp_int < torch.tensor(len(sigmas) - 1, device=t_steps_temp_int.device, dtype=torch.long)
            print(f"mask shape: {mask.shape}, device: {mask.device}")
            print(f"sigma_lower shape: {sigma_lower.shape}, device: {sigma_lower.device}")
            print(f"sigma_upper shape: {sigma_upper.shape}, device: {sigma_upper.device}")
            interpolated = sigma_lower + t_steps_temp_frac * (sigma_upper - sigma_lower)
            t_steps = torch.where(mask, interpolated, sigma_lower)
            print(f"t_steps shape: {t_steps.shape}, device: {t_steps.device}, min: {t_steps.min().item()}, max: {t_steps.max().item()}")
        

        
    else:
        raise ValueError("Got wrong schedule type {}".format(schedule_type))
    
    return t_steps.to(device)


# Copied from the DPM-Solver codebase (https://github.com/LuChengTHU/dpm-solver).
# Different from the original codebase, we use the VE-SDE formulation for simplicity
# while the official implementation uses the equivalent VP-SDE formulation. 
##############################
### Utils for DPM-Solver++ ###
##############################
#----------------------------------------------------------------------------

def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        v: a PyTorch tensor with shape [N].
        dim: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,)*(dims - 1)]
    
#----------------------------------------------------------------------------

def dynamic_thresholding_fn(x0):
    """
    The dynamic thresholding method
    """
    dims = x0.dim()
    p = 0.995
    s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
    s = expand_dims(torch.maximum(s, 1. * torch.ones_like(s).to(s.device)), dims)
    x0 = torch.clamp(x0, -s, s) / s
    return x0

#----------------------------------------------------------------------------

def dpm_pp_update(x, model_prev_list, t_prev_list, t, order, predict_x0=True, scale=1):
    if order == 1:
        return dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1], predict_x0=predict_x0, scale=scale)
    elif order == 2:
        return multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, predict_x0=predict_x0, scale=scale)
    elif order == 3:
        return multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, predict_x0=predict_x0, scale=scale)
    else:
        raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

#----------------------------------------------------------------------------

def dpm_solver_first_update(x, s, t, model_s=None, predict_x0=True, scale=1):
    s, t = s.reshape(-1, 1, 1, 1), t.reshape(-1, 1, 1, 1)
    lambda_s, lambda_t = -1 * s.log(), -1 * t.log()
    h = lambda_t - lambda_s

    phi_1 = torch.expm1(-h) if predict_x0 else torch.expm1(h)
    if predict_x0:
        x_t = (t / s) * x - scale * phi_1 * model_s
    else:
        x_t = x - scale * t * phi_1 * model_s
    return x_t

#----------------------------------------------------------------------------

def multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, predict_x0=True, scale=1):
    t = t.reshape(-1, 1, 1, 1)
    model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
    t_prev_1, t_prev_0 = t_prev_list[-2].reshape(-1, 1, 1, 1), t_prev_list[-1].reshape(-1, 1, 1, 1)
    lambda_prev_1, lambda_prev_0, lambda_t = -1 * t_prev_1.log(), -1 * t_prev_0.log(), -1 * t.log()

    h_0 = lambda_prev_0 - lambda_prev_1
    h = lambda_t - lambda_prev_0
    r0 = h_0 / h
    D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
    phi_1 = torch.expm1(-h) if predict_x0 else torch.expm1(h)
    if predict_x0:
        x_t = (t / t_prev_0) * x - scale * (phi_1 * model_prev_0 + 0.5 * phi_1 * D1_0)
    else:
        x_t = x - scale * (t * phi_1 * model_prev_0 + 0.5 * t * phi_1 * D1_0)
    return x_t

#----------------------------------------------------------------------------

def multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, predict_x0=True, scale=1):
    
    t = t.reshape(-1, 1, 1, 1)
    model_prev_2, model_prev_1, model_prev_0 = model_prev_list[-3], model_prev_list[-2], model_prev_list[-1]
    
    t_prev_2, t_prev_1, t_prev_0 = t_prev_list[-3], t_prev_list[-2], t_prev_list[-1]
    t_prev_2, t_prev_1, t_prev_0 = t_prev_2.reshape(-1, 1, 1, 1), t_prev_1.reshape(-1, 1, 1, 1), t_prev_0.reshape(-1, 1, 1, 1)
    lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = -1 * t_prev_2.log(), -1 * t_prev_1.log(), -1 * t_prev_0.log(), -1 * t.log()

    h_1 = lambda_prev_1 - lambda_prev_2
    h_0 = lambda_prev_0 - lambda_prev_1
    h = lambda_t - lambda_prev_0
    r0, r1 = h_0 / h, h_1 / h
    D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
    D1_1 = (1. / r1) * (model_prev_1 - model_prev_2)
    D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
    D2 = (1. / (r0 + r1)) * (D1_0 - D1_1)
    
    phi_1 = torch.expm1(-h) if predict_x0 else torch.expm1(h)
    phi_2 = phi_1 / h + 1. if predict_x0 else phi_1 / h - 1.
    phi_3 = phi_2 / h - 0.5
    if predict_x0:
        x_t = (t / t_prev_0) * x - scale * (phi_1 * model_prev_0 - phi_2 * D1 + phi_3 * D2)
    else:
        x_t =  x - scale * (t * phi_1 * model_prev_0 + t * phi_2 * D1 + t * phi_3 * D2)
    return x_t
