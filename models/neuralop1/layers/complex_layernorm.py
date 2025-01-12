import torch
import torch.nn as nn

class MagnitudeLayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super(MagnitudeLayerNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        # x should be a complex tensor
        magnitude = x.abs()  # Compute the magnitude of the complex numbers
        mean = magnitude.mean(dim=-1, keepdim=True)  # Mean of magnitudes
        std = magnitude.std(dim=-1, keepdim=True, unbiased=False)  # Standard deviation of magnitudes

        # Normalize magnitudes
        normalized_magnitude = (magnitude - mean) / (std + self.eps)

        # Apply the normalized magnitudes back to the complex tensor
        # Retain the original phase of the complex numbers
        phase = x.angle()
        normalized_complex = torch.polar(normalized_magnitude, phase)
        
        return normalized_complex
