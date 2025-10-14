import os
import pathlib
from typing import Optional, Union

import hydra
import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import WeightedRandomSampler

from algorithms.diffusion_forcing import (
    DiffusionForcingHumanoid,
    DiffusionForcingHumanoidGoal,
    DiffusionForcingHumanoidObstacle,
    DiffusionForcingHumanoidSteer,
)
from uniphys.datasets import StateActionDataset
from .exp_base import BaseLightningExperiment

class IsaacHumanoidExperiment(BaseLightningExperiment):

    compatible_algorithms = dict(
        df_humanoid=DiffusionForcingHumanoid,
        df_humanoid_goal=DiffusionForcingHumanoidGoal,
        df_humanoid_steer=DiffusionForcingHumanoidSteer,
        df_humanoid_obstacle=DiffusionForcingHumanoidObstacle,
    )

    compatible_datasets = dict(
        isaac_babel=StateActionDataset,
    )

    def create_player(self, player):
        self.player = player

        ### freeze PULSE / PHC
        for param in self.player.model.parameters():
            param.requires_grad = False

        self.player.has_batch_dimension = True
        self.env = player.env
        if not self.algo:
            self.algo = self._build_algo()

        self.algo.player = self.player
        self.algo.env = self.env

    def _build_training_loader(self) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        self.train_dataset = self._build_dataset("training")
        if self.cfg.training.train_key_action is not None:
            assert len(self.cfg.training.train_key_action) == len(self.cfg.training.train_action_sample_prob)
            # sampling_prob = [prob / sum(self.cfg.training.train_action_sample_prob) for prob in self.cfg.training.train_action_sample_prob]
            sampling_prob = self.cfg.training.train_action_sample_prob
            sample_weights = self.train_dataset.calculate_weights(keyword_list=self.cfg.training.train_key_action, weighted_sampling_prob=sampling_prob)
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            print("Set up training dataset sampler:")
            print("Action: {}".format(self.cfg.training.train_key_action))
            print("Prob: {}".format(sampling_prob))
            shuffle = False
        else:
            sampler = None
            shuffle = (
                False if isinstance(self.train_dataset, torch.utils.data.IterableDataset) else self.cfg.training.data.shuffle
            )
        if self.train_dataset:
            print("Prepare train dataloader done!")
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.training.data.num_workers),
                shuffle=shuffle,
                sampler=sampler,
                persistent_workers=True,
            )
        
        else:
            return None

    
    def training(self) -> None:
        """
        All training happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.training.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))
        if "checkpointing" in self.cfg.training:
            callbacks.append(
                ModelCheckpoint(
                    pathlib.Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]) / "checkpoints",
                    **self.cfg.training.checkpointing,
                )
            )

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger if self.logger else False,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            callbacks=callbacks,
            gradient_clip_val=self.cfg.training.optim.gradient_clip_val,
            val_check_interval=self.cfg.validation.val_every_n_step,
            limit_val_batches=self.cfg.validation.limit_batch,
            check_val_every_n_epoch=self.cfg.validation.val_every_n_epoch,
            accumulate_grad_batches=self.cfg.training.optim.accumulate_grad_batches,
            precision=self.cfg.training.precision,
            detect_anomaly=False,
            num_sanity_val_steps=int(self.cfg.debug),
            max_epochs=self.cfg.training.max_epochs,
            max_steps=self.cfg.training.max_steps,
            max_time=self.cfg.training.max_time,
        )

        train_dataloader = self._build_training_loader()

        """
        any dataloader post-processing here
        """

        train_data_stats = {"Mean": self.train_dataset.Mean, "Std": self.train_dataset.Std, "ActionMean": self.train_dataset.ActionMean, "ActionStd": self.train_dataset.ActionStd, "zMean": self.train_dataset.zMean, "zStd": self.train_dataset.zStd}
        self.algo.state_mean, self.algo.state_std = self.train_dataset.Mean, self.train_dataset.Std
        self.algo.action_mean, self.algo.action_std = self.train_dataset.ActionMean, self.train_dataset.ActionStd
        self.algo.z_mean, self.algo.z_std = self.train_dataset.zMean, self.train_dataset.zStd

        np.save(os.path.join(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"], "train_data_stats.npy"), train_data_stats)

        trainer.fit(
            self.algo,
            train_dataloaders=train_dataloader,
            ckpt_path=self.ckpt_path,
        )

    @torch.no_grad()
    def interact(self):
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.validation.compile:
            self.algo = torch.compile(self.algo)

        if self.ckpt_path:
            checkpoint = torch.load(self.ckpt_path, map_location="cpu")
            print("restoring from {}".format(self.ckpt_path))
            self.algo.load_state_dict(checkpoint["state_dict"], strict=False)

            stats = np.load(os.path.join(os.path.dirname(os.path.dirname(self.ckpt_path)), "train_data_stats.npy"), allow_pickle=True)[()]
            self.algo.state_mean, self.algo.state_std = stats["Mean"], stats["Std"]
            self.algo.action_mean, self.algo.action_std = stats["ActionMean"], stats["ActionStd"]
            if "zMean" in stats:
                self.algo.z_mean, self.algo.z_std = stats["zMean"], stats["zStd"]

        else:
            raise ValueError("Pretrained checkpoint missing!")
        
        self.algo.diffusion_model.cuda()
        self.algo.diffusion_model.eval()

        self.algo.player = self.player
        self.algo.env = self.player.env
        self.algo.save_dir = os.path.dirname(os.path.dirname(self.ckpt_path))
        self.algo.resume_step = checkpoint["global_step"] if checkpoint is not None else 0
        self.algo.interact(save=False)
    
    
    def evaluate_t2m_babel(self):

        #### Prepare model
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.validation.compile:
            self.algo = torch.compile(self.algo)

        if self.ckpt_path:
            checkpoint = torch.load(self.ckpt_path, map_location="cpu")
            print("restoring from {}".format(self.ckpt_path))
            self.algo.load_state_dict(checkpoint["state_dict"], strict=False)
            # checkpoint = None

            stats = np.load(os.path.join(os.path.dirname(os.path.dirname(self.ckpt_path)), "train_data_stats.npy"), allow_pickle=True)[()]
            self.algo.state_mean, self.algo.state_std = stats["Mean"], stats["Std"]
            self.algo.action_mean, self.algo.action_std = stats["ActionMean"], stats["ActionStd"]
            if "zMean" in stats:
                self.algo.z_mean, self.algo.z_std = stats["zMean"], stats["zStd"]

        else:
            raise NotImplementedError

        self.algo.diffusion_model.cuda()
        self.algo.diffusion_model.eval()

        self.algo.player = self.player  ## Expert pulse player
        self.algo.env = self.player.env
        """
        setting for env  [todo: move to config]
        """
        self.algo.env.task._termination_distances[:] = 1000000   # large termination distances for the sampling mode
        self.algo.env.task.termination_mode = "sampling"

        self.algo.save_dir = os.path.dirname(os.path.dirname(self.ckpt_path))
        self.algo.resume_step = checkpoint["global_step"] if checkpoint is not None else 0 #int(self.ckpt_path.split("=")[-1].split(".")[0])

        self.algo.evaluate_t2m_babel(save=True)
