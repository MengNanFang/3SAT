# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.byol import byol_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params
from solo.utils.rtcl_kmeans import   kmeans
from torch.linalg import norm
import math

class SAT(BaseMomentumMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements BYOL (https://arxiv.org/abs/2006.07733).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of projected features.
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
        """

        super().__init__(cfg)

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        pred_hidden_dim: int = cfg.method_kwargs.pred_hidden_dim


        self.num_clusters = cfg.adv.num_clusters
        self.warm_up_stage = cfg.adv.warm_up_stage
        # self.warm_up_stage = 1
        self.mixup_alpha = cfg.adv.mixup_alpha
        self.cluster_centers:torch.Tensor= None

        self.pseudo_classifier: nn.Module = nn.Linear(self.features_dim, self.num_clusters)

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(SAT, SAT).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_hidden_dim")

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"name": "projector", "params": self.projector.parameters()},
            {"name": "predictor", "params": self.predictor.parameters()},
            # {"name": "pseudo_classifier",
            #  "params": self.pseudo_classifier.parameters(),
            #  "lr": self.classifier_lr,
            #     "weight_decay": 0,},
        ]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        out.update({"z": z, "p": p})
        return out
    


    def adv_forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.backbone(X)
        z = self.projector(feats)
        p = self.predictor(z)
        return p  
    
    def cl_adv_forward(self, X: torch.Tensor) -> Dict[str, Any]:

        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.backbone(X)
        z = self.projector(feats)
        p = self.predictor(z)
        return p

    def pgd_attack(self,images, labels, eps=8. / 255., alpha=2. / 255., iters=5, randomInit=True):
        loss = byol_loss_func

        # init
        if randomInit:
            delta = torch.rand_like(images) * eps * 2 - eps
        else:
            delta = torch.zeros_like(images)
        delta = torch.nn.Parameter(delta, requires_grad=True)

        for i in range(iters):
 
            outputs = self.adv_forward(images + delta)
            self.backbone.zero_grad()
            self.projector.zero_grad()
            self.predictor.zero_grad()
            cost = loss(outputs, labels)
            # cost.backward()
            delta_grad = torch.autograd.grad(cost, [delta])[0]

            delta.data = delta.data + alpha * delta_grad.sign()
            delta.grad = None
            delta.data = torch.clamp(delta.data, min=-eps, max=eps)
            delta.data = torch.clamp(images + delta.data, min=0, max=1) - images

        self.backbone.zero_grad()
        self.projector.zero_grad()
        self.predictor.zero_grad()

        return (images + delta).detach()
    
    def weight_adv(self):
        w = min(1.0,  self.current_epoch // self.warm_up_stage ) * ((self.current_epoch -self.warm_up_stage ) / (self.max_epochs -self.warm_up_stage ) )
        self.log("weight_adv",w)
        return w
    
    def exp_weight_schedule(self,current_epoch, max_epochs=1000,a = 2):
        """
        Extended exponential scheduling function that increases from 0 to 1 in a convex manner
        over an extended range of x from 0 to x_max.
        
        Args:
        x (float): A value between 0 and x_max inclusive.
        a (float): A positive constant to adjust the steepness of the curve.
        x_max (float): The maximum value of x, defaults to 1000.

        Returns:
        float: The output of the adjusted exponential function.
        """
        if 0 <= current_epoch <= max_epochs:
            scaled_x = current_epoch / max_epochs  # Scale x to be between 0 and 1
            w = (math.exp(a * scaled_x) - 1) / (math.exp(a) - 1)
            self.log("weight_adv",w)
            return w
        else:
            raise ValueError(f"Input x should be between 0 and {max_epochs}.")

    def mixup_target(self, target, lam=1.):
        y1 = target
        y2 = target.flip(0)
        return y1 * lam + y2 * (1. - lam)

    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass for the multicrop views.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[]: a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().multicrop_forward(X)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        out.update({"z": z, "p": p})
        return out
    
    def on_after_backward(self):
  
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
        
            total_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = norm(p.grad)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            self.logger.experiment.add_scalar('Gradient_Norm', total_norm, self.trainer.global_step)

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict:
        """Performs the forward pass of the momentum backbone and projector.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of
                the parent and the momentum projected features.
        """

        out = super().momentum_forward(X)
        z = self.momentum_projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for RTCL reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of BYOL and classification loss.
        """
        _, X, targets = batch
        # ori_x = X[-1]
        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        Z = out["z"]
        P = out["p"]
        Z_momentum = out["momentum_z"]

        # ------- negative consine similarity loss -------
        neg_cos_sim = 0
        for v1 in range(self.num_large_crops-1):
            for v2 in np.delete(range(self.num_crops), v1):
                neg_cos_sim += byol_loss_func(P[v2], Z_momentum[v1])

    # calculate std of features
        with torch.no_grad():
            z_std = F.normalize(torch.stack(Z[: self.num_large_crops]), dim=-1).std(dim=1).mean()

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
    # Calculating Adversarial Loss
        adv_x = self.pgd_attack(X[2],Z_momentum[2])
        adv_p = self.cl_adv_forward(adv_x)
        
        adv_loss = byol_loss_func(adv_p,Z_momentum[2])
        self.log("adv_loss",adv_loss)


        weight = self.weight_adv()
        return  neg_cos_sim + class_loss  +  weight*adv_loss 