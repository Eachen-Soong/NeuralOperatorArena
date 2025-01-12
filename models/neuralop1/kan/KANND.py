from . import KAN
import torch

class KANND(KAN):
    def forward(self, x:torch.Tensor):
        # Assume the input shape is: [B, C, S1, S2, ...]
        out_shape = list(x.shape)
        out_shape[1] = self.width[-1][0]+self.width[-1][1]
        x = x.view((-1, self.width[0][0]+self.width[0][1]))
        out = super().forward(x)
        # print(type(out), len(out))
        return out.view(out_shape)

