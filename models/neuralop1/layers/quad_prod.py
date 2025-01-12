from torch import nn
import torch
import numpy as np
from .mlp import MLP

def get_mat_index(index, column):
    if (index+1)% column == 0:
        i = (index+1)//column - 1
        j = column - 1
    else:
        i = (index+1)//column
        j = ((index+1)%column) - 1
    if i>j:
        tmp = i;    i = j;    j = tmp
    return torch.tensor([i, j])

class QuadPath(nn.Module):
    def __init__(self, in_dim, out_dim, num_quad, num_prod, skip=True):
        super(QuadPath, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_quad = num_quad
        self.num_prod = num_prod
        self.out_range = torch.tensor(np.array(range(out_dim)), dtype=torch.int16)
        
        self.prep_stage = True
        self.skip = skip
        # Stage 1 params
        self.quadratic = nn.Conv2d(self.in_dim, self.num_quad, 1)  # First transformation for quadratic
        self.quadratic.weight.data = self.gram_schmidt(self.quadratic.weight.data.squeeze()).unsqueeze(-1).unsqueeze(-1)  # Initialize weights orthogonally
        self.quad_linear = nn.Conv2d(self.num_quad, self.out_dim, 1)  # Second transformation for quadratic
        # Stage 2 params
        self.prod_indices = torch.zeros([num_prod, 2], dtype=torch.int16)  # quad_indices[0] <= ~[1]
        self.prod_linear = nn.Conv2d(num_prod, self.out_dim, 1)
        nn.init.xavier_uniform_(self.quad_linear.weight, 1)

    def forward(self, x):
        if self.prep_stage:
            y_quadratic = self.quadratic(x)
            out_quadratic = torch.square(y_quadratic)
            out_quadratic = self.quad_linear(out_quadratic)
            if self.skip:
                out_quadratic += x
        else:
            out_quadratic = self.calc_quads(x)
            out_quadratic = self.prod_linear(out_quadratic)
            if self.skip:
                out_quadratic += x

        return out_quadratic
    
    def calc_quads(self, x):
        # out_quadratic = torch.stack([func(x_T) for func in self.significant_quad_funcs]).T
        # x.shape = [B, C]
        out_quadratic = torch.stack([x[..., self.prod_indices[i, 0]] * x[..., self.prod_indices[i, 1]] for i in self.out_range], dim=-1)
        return out_quadratic
    
    def get_mat(self, linear):
        return linear.weight.data

    # replace previous representations if the proportion of new representation reaches threshold
    def replace_previous(self, threshold=0.0):
        Q = self.get_mat(self.quadratic)
        Lams = self.get_mat(self.quad_linear)
        n = Lams.shape[0]
        # A = torch.einsum('a b , n b c ->n a c', Q.T, Lams.diag_embed())
        # A = torch.einsum('n a b , b c ->n a c', A, Q)
        A = Q.T @ Lams.diag_embed() @ Q
        B = torch.stack([torch.triu(2. * A[i] - torch.diag(A[i].diag())) for i in range(n)])
        B1 = torch.zeros(B.shape)
        P = torch.zeros(B.shape)

        # Pick the top num_prod for each matrix        
        for i in range(n):
            B_view = B[i].view(-1)
            indices = torch.topk(abs(B_view), self.num_prod)
            sum_prods = torch.sum(B_view)
            for j in range(self.num_prod):
                idx = indices[1][j]
                index = get_mat_index(idx, A.shape[1])
                B1[i, index[0], index[1]] = B_view[idx]
                P[i, index[0], index[1]] = B_view[idx] / sum_prods

        # Pick top num_prod elems from P
        P_view = torch.sum(P, dim=0).view(-1)
        indices = torch.topk(P_view, self.num_prod)
        P1 = torch.zeros(B.shape)
        indices_2d = torch.zeros([self.num_prod, 2], dtype=torch.int16)
        for j in range(self.num_prod):
            idx = indices[1][j]
            index = get_mat_index(idx, A.shape[1])
            indices_2d[j] = index
            P1[:, index[0], index[1]] = P[:, index[0], index[1]]
        
        proportion = torch.sum(P1, dim=[1, 2])
        if_pass = torch.all(proportion >= threshold)
        
        if not if_pass:
            return proportion
        
        self.prep_stage = False

        self.prod_indices = indices_2d

        for (j, idx) in enumerate(indices_2d):
            self.prod_linear.weight.data[:, j] = B[:, idx[0], idx[1]]

        # Update the initial matrices
        # A_new = A.clone()
        # for idx in self.quad_indices:
        #     A_new[idx[0], idx[1]] = 0
        #     A_new[idx[1], idx[0]] = 0

        # L, V = torch.linalg.eig(A_new)

        # self.quadratic.weight.data = V.real.clone()[:self.num_quad, :]
        # self.quad_linear_prep.weight.data = L[:self.num_quad_prep].real.clone()
        return proportion
    
    def normalize(self):
        """
        Normalizes the rows of the quadratic layer and adjusts the quad_linear_prep accordingly.
        """
        with torch.no_grad():
            norms = torch.norm(self.quadratic.weight.data, dim=1, keepdim=True)
            self.quadratic.weight.data /= norms
            self.quad_linear.weight.data *= norms.squeeze()

    def gram_schmidt(self, v):
        """
        Applies the Gram-Schmidt method to the given matrix v for QR Factorization.
        The orthogonalization is performed along the rows, not the columns.
        """
        ortho_matrix = torch.zeros_like(v)
        for i in range(v.size(0)):
            # orthogonalization
            vec = v[i, :]
            space = ortho_matrix[:i, :]
            projection = torch.mm(space, vec.unsqueeze(1))
            vec = vec - torch.sum(projection, dim=0)
            # normalization
            norm = torch.norm(vec)
            vec = vec / norm
            ortho_matrix[i, :] = vec
        return ortho_matrix

    def reorder(self):
        """
        Reorders the rows of the quadratic layer based on the sum of the absolute values of the weights in each row of quad_linear_prep.
        """
        with torch.no_grad():
            # Get the indices that would sort the sum of the absolute values of quad_linear_prep's weights
            _, sorted_indices = torch.sort(self.quad_linear.weight.abs().sum(dim=0), descending=True)
            self.quadratic.weight.data = self.quadratic.weight.data[sorted_indices]
            self.quad_linear.weight.data = self.quad_linear.weight.data[:, sorted_indices]

    def orthonormalize_prep(self):
        self.normalize()
        self.reorder()
        self.gram_schmidt(self.quadratic.weight.data)

    def orthogonality_loss(self):
        """
        A loss function that encourages the weights of the quadratic layer to be orthogonal.
        """
        w = self.quadratic.weight
        gram = torch.mm(w, w.t())
        eye = torch.eye(gram.shape[0], device=gram.device)
        loss = torch.norm(gram - eye)
        return loss


class ProductLayer(nn.Module):
    def __init__(self, in_dim, num_prods, out_dim, n_dim=2):
        super(ProductLayer, self).__init__()
        self.in_dim = in_dim
        self.num_prods = num_prods
        self.out_dim = out_dim
        assert in_dim >= 2*num_prods, "Error: in_dim < 2*num_prods!"
        self.in_dim_linear = in_dim + num_prods
        # self.linear = nn.Linear(self.in_dim_linear, self.out_dim, bias=False)
        self.linear = MLP(self.in_dim_linear, self.out_dim, n_layers=1, n_dim=n_dim)
        self.range_prods = torch.tensor(np.array(range(self.num_prods)), dtype=int)

    def get_prods(self, x):
        return torch.stack([x[:, 2 *i, ...] * x[:, 2 *i + 1, ...] for i in self.range_prods], dim=1)

    def forward(self, x):
        x = torch.cat((x, self.get_prods(x)), dim=1)
        x = self.linear(x)
        return x
    

class ProductPath(nn.Module):
    def __init__(self, in_dim, num_prod):
        super(ProductPath, self).__init__()
        self.in_dim = in_dim
        self.num_prods = num_prod
        assert in_dim >= 2*num_prod, "Error: in_dim < 2*num_prods!"
        self.in_dim_linear = in_dim + num_prod
        self.range_prods = torch.tensor(np.array(range(self.num_prods)), dtype=int)

    def forward(self, x):
        return torch.stack([x[:, 2 *i, ...] * x[:, 2 *i + 1, ...] for i in self.range_prods], dim=1)

class ProductGating(nn.Module):
    def __init__(self, in_dim, num_prod, clamp_thresh=16.):
        super(ProductGating, self).__init__()
        self.in_dim = in_dim
        self.num_prod = num_prod
        assert in_dim >= 2*num_prod, "Error: in_dim < 2*num_prod!"
        self.in_dim_linear = in_dim + num_prod
        self.range_prod = torch.arange(0, num_prod, 1, dtype=torch.int)
        # TODO: more initialization!
        # gating_coeff.size: [num_prod] (the first dimension for broadcast on batches)
        self.gating_coeff = 0.5 * torch.ones([num_prod])
        self.clamp_thresh = nn.Parameter(torch.tensor(clamp_thresh))

    def forward(self, x):
        prods = torch.stack([x[:, 2 * i, ...] * x[:, 2 * i + 1, ...] for i in self.range_prod], dim=1)
        prods = torch.clamp(prods, min=-self.clamp_thresh, max=self.clamp_thresh)
        coeff = torch.clamp(self.gating_coeff, min=0, max=1).view(1, -1, *([1] * (x.dim() - 2))).to(x.device)

        tmp = prods * coeff
        tmp = tmp + x[:, -self.num_prod:, ...] * (1. - coeff)
        new_x = x.clone()
        new_x[:, -self.num_prod:, ...] = tmp
        return new_x

# class ProductLayer2D(ProductLayer):
#     def __init__(self, in_dim, num_prods, out_dim):
#         super().__init__(in_dim, num_prods, out_dim)
#         # self.linear = nn.Conv2d(self.in_dim_linear, self.out_dim, 1)
#         self.range_prods = torch.tensor(np.array(range(self.num_prods)), dtype=int)
    
#     def get_prods(self, x):
#         return torch.stack([x[:, 2 *i, ...] * x[:, 2 *i + 1, ...] for i in self.range_prods], dim=1)
    
#     def forward(self, x):
#         x = torch.cat((x, self.get_prods(x)), dim=1)
#         x = self.linear(x)
#         return x