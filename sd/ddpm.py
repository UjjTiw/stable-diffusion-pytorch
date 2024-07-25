import torch
import numpy as np
# Beta scheduler amount of noise at each timestep
class DDPMsampler:
    def __init__(self, generator: torch.Generator, num_training_steps = 1000, beta_start:float=0.00085, bet_end: float=0.0120):
        self.beta = torch.linspace(beta_start**0.5, bet_end**0.5, num_training_steps, dtype=torch.float32)**2
        self.alpha = 1- self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, 0) # [alpha_0, alpha_0 * alpha_1, alhpa_0 * alpha_1*alpha_2]
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps =num_inference_steps

        step_ratio =self.num_training_steps//self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps)*step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps= torch.from_numpy(timesteps)

    def __get_previous_timestep(self, timestep:int):
        prev_t = timestep - (self.num_training_steps//self.num_inference_steps)
        return prev_t
    
    def _get_var(self, timestep: int) -> torch.Tensor:
        prev_t = self.__get_previous_timestep(timestep)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >=0 else self.one
        current_beta_t = 1 - alpha_prod_t/alpha_prod_t_prev

        variance = (1- alpha_prod_t_prev) / (1-alpha_prod_t) * current_beta_t

        variance = torch.clamp(variance, min=1e-20)

        return variance


    def step(self, timestep: int, latents: torch.Tensor , model_output:torch.Tensor):  # latents -> xt, model_output-> epsilon
        t = timestep
        prev_t = self.__get_previous_timestep(t)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev= self.alpha_cumprod[prev_t] if prev_t >=0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1- alpha_prod_t_prev
        current_alpha_t = alpha_prod_t/alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        
        # Compute the predicted original sample using formula 15
        pred_original_sample = (latents - (beta_prod_t**0.5) * model_output)/ alpha_prod_t ** 0.5

        # Compute the coefficient for pred_original_sample and current sample x_t
        pred_original_sample_coef = (alpha_prod_t_prev **0.5 * current_beta_t)/beta_prod_t
        current_sample_coef = (current_alpha_t**0.5)*(beta_prod_t_prev)/beta_prod_t 
        
        # Compute the predicted sample mean 
        pred_prev_sample = pred_original_sample_coef * pred_original_sample + current_sample_coef * latents
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            variance = (self._get_var(t) ** 0.5) * noise
        
        


    def add_noise(self, original_sample:torch.FloatTensor, timesteps: torch.IntTensor)->torch.FloatTensor:
        alpha_cumprod = self.alpha_cumprod.to(device=original_sample.device, dtype=original_sample.dtype)
        timesteps = timesteps.to(original_sample.device)

        sqrt_alpha_prod = alpha_cumprod(timesteps) ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod) < len(original_sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_sample.shape):
            sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noise = torch.randn(original_sample.shape, generator=self.generator, device=original_sample.device, device=original_sample.device, dtype=original_sample.dtype)
        noisy_sample = (sqrt_alpha_prod * original_sample) + (sqrt_one_minus_alpha_prod) * noise
        return noisy_sample 
    
