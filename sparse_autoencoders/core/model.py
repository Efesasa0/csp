"""SAE model definitions: SparseAutoencoder and TopKSAE."""

import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(self, n_input=2048, n_hidden=16384):
        super().__init__()
        self.encoder = nn.Linear(n_input, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_input, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(n_input))
        self.relu = nn.ReLU()

    def forward(self, x):
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        x_cent = x - self.b_dec
        z = self.relu(self.encoder(x_cent))
        x_hat = self.decoder(z) + self.b_dec
        if squeeze:
            x_hat = x_hat.squeeze(0)
            z = z.squeeze(0)
        return x_hat, z


class TopKSAE(nn.Module):
    def __init__(self, n_input=2048, n_hidden=16384, k=32):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(n_input, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_input, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(n_input))

        self.decoder.weight.data = self.encoder.weight.data.t().clone()
        self.decoder.weight.data /= (
            self.decoder.weight.data.norm(dim=0, keepdim=True) + 1e-8
        )

    def forward(self, x):
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        x_cent = x - self.b_dec
        latents = torch.relu(self.encoder(x_cent))

        if self.k < latents.shape[-1]:
            topk_vals, _ = torch.topk(latents, self.k, dim=-1)
            threshold = topk_vals[..., [-1]]
            latents = torch.where(
                latents >= threshold, latents, torch.zeros_like(latents)
            )

        x_hat = self.decoder(latents) + self.b_dec

        if squeeze:
            x_hat = x_hat.squeeze(0)
            latents = latents.squeeze(0)

        return x_hat, latents
