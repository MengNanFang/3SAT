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


class RTCL(BaseMomentumMethod):
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

        cfg = super(RTCL, RTCL).add_and_assert_specific_cfg(cfg)

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
            {"name": "pseudo_classifier",
             "params": self.pseudo_classifier.parameters(),
             "lr": self.classifier_lr,
                "weight_decay": 0,},
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
    
    def linear_cons_forward(self, X: torch.Tensor) -> torch.Tensor:

        out = self.backbone(X)
        z = self.projector(out)
        p = self.predictor(z)
        return p
    
    def pesudo_forward(self, feats: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        pseudo_logits  = self.pseudo_classifier(feats.detach()) 
        return pseudo_logits
    
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
        pseudo_logits  = self.pseudo_classifier(feats) 
        return pseudo_logits    
    
    def cl_adv_forward(self, X: torch.Tensor) -> Dict[str, Any]:

        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.backbone(X)
        z = self.projector(feats)
        p = self.predictor(z)
        return p

    def  pesudo_lables(self, feats:torch.Tensor):
        ## using  KNN  produce pesudo labels
        ptg, self.cluster_centers = kmeans(feats.detach(),self.num_clusters,distance="cosine",cluster_centers=self.cluster_centers,iter_limit=10,device="cuda:0")
        return ptg
    
    def mixup_target(self, target, lam=1.):
        y1 = target
        y2 = target.flip(0)
        return y1 * lam + y2 * (1. - lam)
    
    def pgd_attack(self,images, labels, eps=8. / 255., alpha=2. / 255., iters=5, randomInit=True):
    # images = images.to(device)
    # labels = labels.to(device)
        loss = nn.CrossEntropyLoss()

        # init
        if randomInit:
            delta = torch.rand_like(images) * eps * 2 - eps
        else:
            delta = torch.zeros_like(images)
        delta = torch.nn.Parameter(delta, requires_grad=True)

        for i in range(iters):
 
            outputs = self.adv_forward(images + delta)
            self.backbone.zero_grad()
            self.pseudo_classifier.zero_grad()
            cost = loss(outputs, labels)
            # cost.backward()
            delta_grad = torch.autograd.grad(cost, [delta])[0]

            delta.data = delta.data + alpha * delta_grad.sign()
            delta.grad = None
            delta.data = torch.clamp(delta.data, min=-eps, max=eps)
            delta.data = torch.clamp(images + delta.data, min=0, max=1) - images

        self.backbone.zero_grad()
        self.pseudo_classifier.zero_grad()

        return (images + delta).detach()
    
    def weight_adv(self):
        w = min(1.0,  self.current_epoch // self.warm_up_stage ) * ((self.current_epoch -self.warm_up_stage ) / (self.max_epochs -self.warm_up_stage ) )
        self.log("weight_adv",w)
        return w
    
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
    
    def on_after_backward(self):
        # 在每次反向传播之后调用
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            # 计算所有梯度的范数
            total_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = norm(p.grad)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            # 使用TensorBoard记录梯度范数
            self.logger.experiment.add_scalar('Gradient_Norm', total_norm, self.trainer.global_step)

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
        # ------- pseudo classification loss -------
        ## using k-means to generate pesudo labels
        pgt = self.pesudo_lables(out["momentum_feats"][2])
        ## Calculating Adversarial Pertibation 
        pseudo_logits = self.pesudo_forward(out["feats"][2])
        pseudo_loss =  F.cross_entropy(pseudo_logits, pgt, ignore_index=-1)
        adv_x = self.pgd_attack(X[2],pgt)
        adv_p = self.cl_adv_forward(adv_x)
        ## Calculating Adversarial Loss
        adv_loss = byol_loss_func(adv_p,Z_momentum[2])
        self.log("adv_loss",adv_loss)
        
        ## Adding Linear Constraint
        # lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        # x =  X[2]
        # x_flipped = x.flip(0).mul_(1. - lam_mix)
        # x.mul_(lam_mix).add_(x_flipped)
         
        # mix_p = self.linear_cons_forward(x)
        # mix_target = self.mixup_target(P[2],lam_mix)
        # mix_p_norm = F.normalize(mix_p,dim=-1)
        # mix_target_norm = F.normalize(mix_target,dim=-1)
        # linear_cons_loss = F.mse_loss(mix_p_norm,mix_target_norm,reduction="mean")

        # return neg_cos_sim + class_loss  + pseudo_loss + self.weight_adv()*adv_loss +  linear_cons_loss
        return neg_cos_sim + class_loss  + pseudo_loss + self.weight_adv()*adv_loss 