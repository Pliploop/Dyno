import torch
from typing import Optional, Tuple, Union

# assumes these are available in your env like in diffusers
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, DDPMSchedulerOutput
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchEulerDiscreteSchedulerOutput,
)
from diffusers.utils.torch_utils import randn_tensor


class DDPMSchedulerPlusPlus(DDPMScheduler):
    """
    DDPMScheduler + CFG++ support via `step_cfgpp`.

    CFG++ rule here:
      - pred_original_sample (x0_hat) is computed from `model_output_guided`
      - prev_sample update mean is computed using `model_output_uncond` (renoising/update term)

    This matches the DDPM mean form:
        mu = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps)
    where eps is taken from the unconditional branch for CFG++.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfgpp = True

    def _model_output_to_epsilon(
        self, model_output: torch.Tensor, sample: torch.Tensor, alpha_prod_t: torch.Tensor, beta_prod_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert model_output (epsilon / v / sample) to epsilon for the current timestep.
        """
        pred_type = self.config.prediction_type
        if pred_type == "epsilon":
            return model_output
        elif pred_type == "v_prediction":
            # v = sqrt(a_bar)*eps - sqrt(1-a_bar)*x0? (diffusers uses:
            # x0 = sqrt(a_bar)*x_t - sqrt(1-a_bar)*v  => v = (sqrt(a_bar)*x_t - x0)/sqrt(1-a_bar)
            # Here we want eps. Using relationship:
            # x0 = sqrt(a_bar)*x_t - sqrt(1-a_bar)*v
            # eps = (x_t - sqrt(a_bar)*x0) / sqrt(1-a_bar)
            # Substitute x0: eps = (x_t - sqrt(a_bar)*(sqrt(a_bar)*x_t - sqrt(1-a_bar)*v)) / sqrt(1-a_bar)
            #             = (x_t - a_bar*x_t + sqrt(a_bar)*sqrt(1-a_bar)*v)/sqrt(1-a_bar)
            #             = sqrt(1-a_bar)*x_t/sqrt(1-a_bar) + sqrt(a_bar)*v
            #             = sqrt(1-a_bar)*x_t / (1?) wait carefully:
            # (1 - a_bar) x_t / sqrt(1-a_bar) = sqrt(1-a_bar)*x_t
            # => eps = sqrt(1-a_bar)*x_t + sqrt(a_bar)*v
            return (beta_prod_t.sqrt() * sample) + (alpha_prod_t.sqrt() * model_output)
        elif pred_type == "sample":
            # model_output is x0; derive eps from x_t = sqrt(a_bar) x0 + sqrt(1-a_bar) eps
            return (sample - alpha_prod_t.sqrt() * model_output) / beta_prod_t.sqrt()
        else:
            raise ValueError(f"Unsupported prediction_type: {pred_type}")

    def _epsilon_to_pred_x0(
        self, eps: torch.Tensor, sample: torch.Tensor, alpha_prod_t: torch.Tensor, beta_prod_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute x0_hat from epsilon (same as base scheduler's epsilon branch).
        """
        return (sample - beta_prod_t.sqrt() * eps) / alpha_prod_t.sqrt()

    def step_cfgpp(
        self,
        model_output_guided: torch.Tensor,
        model_output_uncond: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple[torch.Tensor, torch.Tensor]]:
        """
        CFG++ step for DDPM.

        Args:
            model_output_guided: interpolated output (used to compute x0_hat)
            model_output_uncond: unconditional output (used for update mean)
            timestep: current discrete timestep (int index into alphas_cumprod)
            sample: x_t
        """
        t = timestep
        prev_t = self.previous_timestep(t)

        # handle learned variance heads consistently with base scheduler
        predicted_variance = None
        if model_output_guided.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output_guided, predicted_variance = torch.split(model_output_guided, sample.shape[1], dim=1)
        if model_output_uncond.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output_uncond, _ = torch.split(model_output_uncond, sample.shape[1], dim=1)

        # 1) scalars
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2) Convert both outputs to epsilon form
        eps_guided = self._model_output_to_epsilon(model_output_guided, sample, alpha_prod_t, beta_prod_t)
        eps_uncond = self._model_output_to_epsilon(model_output_uncond, sample, alpha_prod_t, beta_prod_t)

        # 3) x0_hat from guided epsilon (CFG++ denoising / Tweedie estimate)
        pred_original_sample = self._epsilon_to_pred_x0(eps_guided, sample, alpha_prod_t, beta_prod_t)

        # 4) clip/threshold x0_hat (same logic as base)
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)

        # 5) CFG++ update mean using unconditional epsilon (renoising/update term)
        # DDPM posterior mean in epsilon form:
        #   mu = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps)
        # where alpha_t == current_alpha_t, alpha_bar_t == alpha_prod_t
        mu = (sample - (current_beta_t / beta_prod_t.sqrt()) * eps_uncond) / current_alpha_t.sqrt()

        # 6) Add noise (same as base scheduler)
        variance = 0
        if t > 0:
            device = mu.device
            variance_noise = randn_tensor(mu.shape, generator=generator, device=device, dtype=mu.dtype)

            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
            elif self.variance_type == "learned_range":
                var = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * var) * variance_noise
            else:
                variance = (self._get_variance(t, predicted_variance=predicted_variance).sqrt()) * variance_noise

        pred_prev_sample = mu + variance

        if not return_dict:
            return (pred_prev_sample, pred_original_sample)

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)


class FlowMatchEulerDiscretePlusPlus(FlowMatchEulerDiscreteScheduler):
    """
    FlowMatchEulerDiscreteScheduler + CFG++ support via `step_cfgpp`.

    For FM Euler:
      - x0_hat is computed from `model_output_guided` as: x0 = x - sigma * v
      - deterministic update uses unconditional velocity v_u:
            x_{next} = x + dt * v_u
      - stochastic_sampling path keeps using x0_hat (guided) in the convex combination with noise,
        since the "renoising" term is explicit noise (not model_output) in this implementation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfgpp = True

    def step_cfgpp(
        self,
        model_output_guided: torch.FloatTensor,
        model_output_uncond: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        per_token_timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple[torch.FloatTensor]]:
        # keep the same timestep validation as base
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                "Pass a value from `scheduler.timesteps` (float tensor), not integer indices."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast like base
        sample_f32 = sample.to(torch.float32)

        if per_token_timesteps is not None:
            per_token_sigmas = per_token_timesteps / self.config.num_train_timesteps

            sigmas = self.sigmas[:, None, None]
            lower_mask = sigmas < per_token_sigmas[None] - 1e-6
            lower_sigmas = lower_mask * sigmas
            lower_sigmas, _ = lower_sigmas.max(dim=0)

            current_sigma = per_token_sigmas[..., None]
            next_sigma = lower_sigmas[..., None]
            dt = current_sigma - next_sigma
        else:
            sigma_idx = self.step_index
            sigma = self.sigmas[sigma_idx]
            sigma_next = self.sigmas[sigma_idx + 1]
            current_sigma = sigma
            next_sigma = sigma_next
            dt = sigma_next - sigma  # (negative)

        # CFG++ denoised estimate uses guided velocity
        # x0_hat = x - sigma * v
        x0_hat = sample_f32 - current_sigma * model_output_guided.to(torch.float32)

        if self.config.stochastic_sampling:
            noise = torch.randn_like(sample_f32, generator=generator)
            prev_sample = (1.0 - next_sigma) * x0_hat + next_sigma * noise
        else:
            # CFG++ update uses unconditional velocity
            prev_sample = sample_f32 + dt * model_output_uncond.to(torch.float32)

        # advance step index like base
        self._step_index += 1

        if per_token_timesteps is None:
            prev_sample = prev_sample.to(model_output_guided.dtype)

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
