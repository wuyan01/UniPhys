"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from typing import Optional
from tqdm import tqdm
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any
from einops import rearrange

from lightning.pytorch.utilities.types import STEP_OUTPUT

from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from .models.diffusion import Diffusion

class DiffusionForcingBase(BasePytorchAlgo):
    def __init__(self, cfg: DictConfig):

        self.cfg = cfg
        self.x_shape = cfg.x_shape
        self.frame_stack = cfg.frame_stack
        self.x_stacked_shape = list(self.x_shape)
        self.x_stacked_shape[0] *= cfg.frame_stack
        self.guidance_scale = cfg.guidance_scale
        self.context_frames = cfg.context_frames
        self.chunk_size = cfg.chunk_size
        self.external_cond_dim = cfg.external_cond_dim
        self.cond_mask_prob = cfg.get("cond_mask_prob", 0.0)
        self.causal = cfg.causal

        self.uncertainty_scale = cfg.uncertainty_scale
        self.timesteps = cfg.diffusion.timesteps
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.clip_noise = cfg.diffusion.clip_noise
        # self.device = "cuda:0"

        self.cfg.diffusion.cum_snr_decay = self.cfg.diffusion.cum_snr_decay ** (self.frame_stack * cfg.frame_skip)

        self.validation_step_outputs = []

        super().__init__(cfg)

    def _build_model(self):
        self.diffusion_model = Diffusion(
            x_shape=self.x_stacked_shape,
            external_cond_dim=self.external_cond_dim,
            is_causal=self.causal,
            cond_mask_prob=self.cond_mask_prob,
            cfg=self.cfg.diffusion,
        )
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)

    def configure_optimizers(self):
        params = tuple(self.diffusion_model.parameters())
        # import ipdb; ipdb.set_trace()
        self.cfg.lr *= 1.0 #self.trainer.num_gpus
        optimizer_dynamics = torch.optim.AdamW(
            params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.optimizer_beta
        )
        return optimizer_dynamics

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.cfg.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.cfg.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.cfg.lr
        else:
            # # linear decay
            if self.cfg.lr_decay == "linear":
                lr_scale = max(0, (self.cfg.max_steps - self.trainer.global_step) / (self.cfg.max_steps - self.cfg.warmup_steps))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * self.cfg.lr

            # Cosine decay
            elif self.cfg.lr_decay == "cosine":
                progress = (self.trainer.global_step - self.cfg.warmup_steps) / float(self.cfg.max_steps - self.cfg.warmup_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = 0.5 * (1 + np.cos(np.pi * progress)) * self.cfg.lr

            elif self.cfg.lr_decay == "cyclic_cosine":
                # cycle_progress = (self.trainer.global_step - self.cfg.warmup_steps) % self.cfg.cycle_steps
                # # Cosine decay within the cycle (with warm up)
                # for pg in optimizer.param_groups:
                #     pg["lr"] = 0.5 * (1 + np.cos(np.pi * cycle_progress / self.cfg.cycle_steps)) * self.cfg.lr

                cycle_progress = self.trainer.global_step % (self.cfg.warmup_steps + self.cfg.cycle_steps)
                # Warm-up phase at the beginning of each cycle
                if cycle_progress < self.cfg.warmup_steps:
                    lr_scale = cycle_progress / float(self.cfg.warmup_steps)
                else:
                    # Cosine decay phase after the warm-up in each cycle
                    decay_progress = cycle_progress - self.cfg.warmup_steps
                    lr_scale = 0.5 * (1 + np.cos(np.pi * decay_progress / self.cfg.cycle_steps))

                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * self.cfg.lr

            
            else:
                raise ValueError("Unsupported lr_decay [linear, cosine, cyclic_cosine]")

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:

        xs, conditions, masks = self._preprocess_batch(batch)

        xs_pred, loss = self.diffusion_model(xs, conditions, noise_levels=self._generate_noise_levels(xs))
        
        loss = self.reweight_loss(loss, masks)

        # log the loss
        if batch_idx % 20 == 0:
            self.log("training/loss", loss, on_step=True, on_epoch=False, sync_dist=True)

        xs = self._unstack_and_unnormalize(xs)
        xs_pred = self._unstack_and_unnormalize(xs_pred)

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict


    def _generate_noise_levels(self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate noise levels for training.
        """
        num_frames, batch_size, *_ = xs.shape  # T, B, D

        noise_levels = torch.randint(0, self.timesteps, (num_frames, batch_size), device=xs.device)

        if masks is not None:
            # for frames that are not available, treat as full noise
            discard = torch.all(~rearrange(masks.bool(), "(t fs) b -> t b fs", fs=self.frame_stack), -1)
            noise_levels = torch.where(discard, torch.full_like(noise_levels, self.timesteps - 1), noise_levels)

        return noise_levels

    def _generate_scheduling_matrix(self, horizon: int):
        if self.cfg.scheduling_matrix == "pyramid":
            return self._generate_pyramid_scheduling_matrix(horizon, self.uncertainty_scale)
        elif self.cfg.scheduling_matrix == "full_sequence":
            return np.arange(self.sampling_timesteps, -1, -1)[:, None].repeat(horizon, axis=1)
        elif self.cfg.scheduling_matrix == "autoregressive":
            return self._generate_pyramid_scheduling_matrix(horizon, self.sampling_timesteps)
        elif self.cfg.scheduling_matrix == "trapezoid":
            return self._generate_trapezoid_scheduling_matrix(horizon, self.uncertainty_scale)
        else:
            raise NotImplementedError


    def _generate_pyramid_scheduling_matrix(self, horizon: int, uncertainty_scale: float):
        height = self.sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = self.sampling_timesteps + int(t * uncertainty_scale) - m

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)

    def _generate_trapezoid_scheduling_matrix(self, horizon: int, uncertainty_scale: float):
        height = self.sampling_timesteps + int((horizon + 1) // 2 * uncertainty_scale)
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range((horizon + 1) // 2):
                scheduling_matrix[m, t] = self.sampling_timesteps + int(t * uncertainty_scale) - m
                scheduling_matrix[m, -t] = self.sampling_timesteps + int(t * uncertainty_scale) - m

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)

    def reweight_loss(self, loss, weight=None):
        loss = rearrange(loss, "t b (fs c) ... -> t b fs c ...", fs=self.frame_stack)
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape) - 1
            weight = rearrange(
                weight,
                "(t fs) b ... -> t b fs ..." + " 1" * expand_dim,
                fs=self.frame_stack,
            )
            loss = loss * weight

        return loss.mean()

    def _preprocess_batch(self, batch):
        xs = batch[0]
        batch_size, n_frames = xs.shape[:2]

        if n_frames % self.frame_stack != 0:
            raise ValueError("Number of frames must be divisible by frame stack size")
        if self.context_frames % self.frame_stack != 0:
            raise ValueError("Number of context frames must be divisible by frame stack size")

        masks = torch.ones(n_frames, batch_size).to(xs.device)
        n_frames = n_frames // self.frame_stack

        if self.external_cond_dim > 0:
            conditions = batch[1]
            conditions = torch.cat([torch.zeros_like(conditions[:, :1]), conditions[:, 1:]], 1)
            conditions = rearrange(conditions, "b (t fs) d -> t b (fs d)", fs=self.frame_stack).contiguous()
        else:
            conditions = [None for _ in range(n_frames)]

        xs = self._normalize_x(xs)
        xs = rearrange(xs, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack).contiguous()

        return xs, conditions, masks

    def _normalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return (xs - mean) / std

    def _unnormalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return xs * std + mean

    def _unstack_and_unnormalize(self, xs):
        xs = rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        return self._unnormalize_x(xs)
