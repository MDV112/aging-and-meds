import os
import numpy as np
import torch
import torch.nn as nn
from data_loader import TorchDataset
import time
from sklearn.metrics import classification_report
from run import Run
from sklearn.model_selection import train_test_split
from extra_functions import BuildCNN
from collections import OrderedDict
from DSU import DistributionUncertainty
from torch.nn.utils import weight_norm

class CosineLoss():
    def __init__(self, weighted_or_not):
        super(CosineLoss, self).__init__()



class ModulatedCosineLoss():
    def __init__(self, weighted_or_not):
        super(ModulatedCosineLoss, self).__init__()


class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
    """

    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        # if (not self.training) or (np.random.random()) > self.p:
        #     return x

        mean = x.mean(dim=[2], keepdim=False)
        std = (x.var(dim=[2], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1)) / std.reshape(x.shape[0], x.shape[1], 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1) + beta.reshape(x.shape[0], x.shape[1], 1)

        return x


class DomainGeneralization():
    def __init__(self):
        super(DomainGeneralization, self).__init__()
        self.DSU = DistributionUncertainty()
    def create_aug(self, V, alpha=0.9):
        """
        This function creates a positive augmentation for every representation in the mini-batch according to mixup
        algorithm.
        :param V: a mini-batch of embeddings (after e1).
        :param alpha: a fraction that controls the "positivity" of the augmentaiton. The closer it is to 1, the more
        likely the embedding to resemble the representation.
        :return: matrix of augmentations in the same size of the mini-batch.
        """
        vec_idx = torch.arange(V.shape[0])
        A = torch.zeros_like(V)
        for j, v in enumerate(V):  # runs over the first dimension which is the number of examples per batch
            lmbda = ( 1 -alpha ) *torch.rand(1).item() + alpha  # setting uniformly distributed r.v in range [alph,1]
            vec_neg = vec_idx[~np.isin(vec_idx, j)]
            perm = torch.randperm(len(vec_neg))
            v_bar = V[perm[0]]
            A[j] = lmbda *v + ( 1 -lmbda ) *v_bar
        return A


    def L_aug(self, Z, phi_A, n=10):
        """
        This function calculates the mutual information (MI) ratio between the representation to its' positive augmentation
        w.r.t. the MI of the representation to its' n negative augmentations.
        :param Z: mini-batch of embeddings (after e1+e2).
        :param phi_A: mini-batch of augmentations' embeddings (after e2).
        :param n: number of negative augmentations per example (must be lower than batch size).
        :return: mean MI ratio across the mini-batch.
        """
        vec_idx = torch.arange(Z.shape[0])
        I_Z_A = torch.tensor(0.0).to(Z.device)
        eps = 10.0 ** -6
        tau = 10.0 ** -2  # todo: maybe normalize inputs of exponents by inputs norm
        if n >= Z.shape[0]:
            n = Z.shape[0] - 1
        for j, pos_pair in enumerate(zip(Z, phi_A), 0):
            z, phi_A_pos = pos_pair
            vec_neg = vec_idx[~np.isin(vec_idx, j)]
            perm = torch.randperm(len(vec_neg))
            v_bar = phi_A[perm[:n]]
            # set v_bar as a matrix with n rows and flattened data
            A = torch.cat((phi_A_pos.flatten().unsqueeze(0), v_bar.view(v_bar.size(0), -1)))
            sim = torch.exp(tau *torch.matmul(A, z.flatten()))
            L = torch.log(sim[0 ] /(eps + torch.sum(sim)))
            if not(torch.isnan(L) or torch.isinf(L)):  # can happen if tau is not enough to lower the exp in sim
                I_Z_A -= L  # NOTICE THE MINUS
        mean_I_Z_A = I_Z_A /len(Z)
        return mean_I_Z_A

    def L_supp(self, Z, domain_tag):
        """
        This function calculates the support loss to minimize domain-specific effects.
        :param Z: mini-batch of embeddings (after e1+e2).
        :param domain_tag: domain labels (age, hospital number etc.).
        :return: support loss as in the paper.
        """
        B_Z_D = torch.tensor(0.0).to(Z.device)
        if (domain_tag is not None) or not(bool(torch.all(torch.isnan(domain_tag)))) or \
                bool(torch.diff(domain_tag).sum() != 0):
            eps = 10.0 ** -6
            tau = 10.0 ** -4
            for j, z_domain_pair in enumerate(zip(Z, domain_tag), 0):
                z, domain = z_domain_pair
                Z_D = Z[domain_tag != domain]
                # todo: should we drop z from Z?
                if Z_D.size()[0] != 0:
                    nom = torch.sum(torch.exp(tau *torch.matmul(Z_D.view(Z_D.size(0), -1), z.flatten())))
                    den = torch.sum(torch.exp(tau *torch.matmul(Z.view(Z.size(0), -1), z.flatten())))
                    L = torch.log(nom / (den + eps))
                    if not(torch.isnan(L) or torch.isinf(L)):
                        B_Z_D -= L
        mean_B_Z_D = B_Z_D /len(Z)
        return mean_B_Z_D

