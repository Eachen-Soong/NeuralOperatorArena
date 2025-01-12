import torch.nn.functional as F
from .utilities import *
from einops import repeat, rearrange

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, fourier_weight=None, gain=1.):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return 

class SpectralConvProd2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, num_prod, fourier_weight=None, gain=1.):
        super(SpectralConvProd2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.num_prod = num_prod

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels- self.num_prod, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels- self.num_prod, self.modes1, self.modes2, dtype=torch.cfloat))
        self.prod_path = ProductPath(in_channels, self.num_prod)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels - self.num_prod,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        out_ft_prod = self.prod_path(out_ft)
        out_ft = torch.cat((out_ft, out_ft_prod), dim=1)
        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class SpectralConvFactorized2d(nn.Module):
    """
    F-FNO's Spectral Convolution, 
    but without the forecast/backcast linear transforms 
    (moved to the FNO Block to make the code more modulized)
    """
    def __init__(self, in_dim, out_dim, n_modes, fourier_weight, mode):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        self.mode = mode

        self.fourier_weight = fourier_weight
        # Can't use complex type yet. See https://github.com/pytorch/pytorch/issues/59998

    def forward(self, x):
        x = rearrange(x, 'b m n i -> b i m n')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        B, I, M, N = x.shape

        # # # Dimesion Y # # #
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        if self.mode == 'full':
            out_ft[:, :, :, :self.n_modes] = torch.einsum(
                "bixy,ioy->boxy",
                x_fty[:, :, :, :self.n_modes],
                torch.view_as_complex(self.fourier_weight[0]))
        elif self.mode == 'low-pass':
            out_ft[:, :, :, :self.n_modes] = x_fty[:, :, :, :self.n_modes]

        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        if self.mode == 'full':
            out_ft[:, :, :self.n_modes, :] = torch.einsum(
                "bixy,iox->boxy",
                x_ftx[:, :, :self.n_modes, :],
                torch.view_as_complex(self.fourier_weight[1]))
        elif self.mode == 'low-pass':
            out_ft[:, :, :self.n_modes, :] = x_ftx[:, :, :self.n_modes, :]

        xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')

        # # Combining Dimensions # #
        x = xx + xy

        x = rearrange(x, 'b i m n -> b m n i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return x

class SpectralConvFactorizedProd2d(nn.Module):
    """
    F-FNO's Spectral Convolution, 
    but without the forecast/backcast linear transforms 
    (moved to the FNO Block to make the code more modulized)
    """
    def __init__(self, in_dim, out_dim, n_modes, fourier_weight, mode):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        self.mode = mode

        self.fourier_weight = fourier_weight
        # Can't use complex type yet. See https://github.com/pytorch/pytorch/issues/59998

    def forward(self, x):
        x = rearrange(x, 'b m n i -> b i m n')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        B, I, M, N = x.shape

        # # # Dimesion Y # # #
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        if self.mode == 'full':
            out_ft[:, :, :, :self.n_modes] = torch.einsum(
                "bixy,ioy->boxy",
                x_fty[:, :, :, :self.n_modes],
                torch.view_as_complex(self.fourier_weight[0]))
        elif self.mode == 'low-pass':
            out_ft[:, :, :, :self.n_modes] = x_fty[:, :, :, :self.n_modes]

        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        if self.mode == 'full':
            out_ft[:, :, :self.n_modes, :] = torch.einsum(
                "bixy,iox->boxy",
                x_ftx[:, :, :self.n_modes, :],
                torch.view_as_complex(self.fourier_weight[1]))
        elif self.mode == 'low-pass':
            out_ft[:, :, :self.n_modes, :] = x_ftx[:, :, :self.n_modes, :]

        xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')

        # # Combining Dimensions # #
        x = xx + xy

        x = rearrange(x, 'b i m n -> b m n i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return x


def skip_connection(
    in_features, out_features, n_dim=2, bias=False, skip_type="soft-gating"
    ):
    """A wrapper for several types of skip connections.
    Returns an nn.Module skip connections, one of  {'identity', 'linear', soft-gating'}

    Parameters
    ----------
    in_features : int
        number of input features
    out_features : int
        number of output features
    n_dim : int, default is 2
        Dimensionality of the input (excluding batch-size and channels).
        ``n_dim=2`` corresponds to having Module2D.
    bias : bool, optional
        whether to use a bias, by default False
    skip_type : {'identity', 'linear', soft-gating'}
        kind of skip connection to use, by default "soft-gating"

    Returns
    -------
    nn.Module
        module that takes in x and returns skip(x)
    """
    if skip_type.lower() == "soft-gating":
        return SoftGating(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            n_dim=n_dim,
        )
    elif skip_type.lower() == "linear":
        return getattr(nn, f"Conv{n_dim}d")(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            bias=bias,
        )
    elif skip_type.lower() == "identity":
        return nn.Identity()
    else:
        raise ValueError(
            f"Got skip-connection type={skip_type}, expected one of"
            f" {'soft-gating', 'linear', 'id'}."
        )

class SoftGating(nn.Module):
    """Applies soft-gating by weighting the channels of the given input

    Given an input x of size `(batch-size, channels, height, width)`,
    this returns `x * w `
    where w is of shape `(1, channels, 1, 1)`

    Parameters
    ----------
    in_features : int
    out_features : None
        this is provided for API compatibility with nn.Linear only
    n_dim : int, default is 2
        Dimensionality of the input (excluding batch-size and channels).
        ``n_dim=2`` corresponds to having Module2D.
    bias : bool, default is False
    """

    def __init__(self, in_features, out_features=None, n_dim=2, bias=False):
        super().__init__()
        if out_features is not None and in_features != out_features:
            raise ValueError(
                f"Got in_features={in_features} and out_features={out_features}"
                "but these two must be the same for soft-gating"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))
        if bias:
            self.bias = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))
        else:
            self.bias = None

    def forward(self, x):
        """Applies soft-gating to a batch of activations"""
        if self.bias is not None:
            return self.weight * x + self.bias
        else:
            return self.weight * x

class FeedForward(nn.Module):
    def __init__(self, dim, factor, ff_weight_norm, n_layers, layer_norm, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True) if i < n_layers - 1 else nn.Identity(),
                nn.LayerNorm(out_dim) if layer_norm and i == n_layers -
                1 else nn.Identity(),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x
    
def get_mat_index(index, column):
    if (index+1)% column == 0:
        i = (index+1)//column - 1
        j = column - 1
    else:
        i = (index+1)//column
        j = ((index+1)%column) - 1
    if i>j:
        tmp = i;    i = j;    j = tmp
    return torch.tensor([i, j], dtype=torch.int16)

class QuadraticLayer(nn.Module):
    def __init__(self, in_dim, num_quad_prep, num_quad, out_dim):
        super(QuadraticLayer, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_quad_prep = num_quad_prep
        self.num_quad = num_quad
        self.linear = nn.Linear(self.in_dim, self.out_dim)  # Linear layer
        self.prep_stage = True
        # Stage 0: only quad_linear_prep; 1: both; 2: only quad_linear
        self.stage = 0
        # Phase 1 parameters
        self.quadratic = nn.Linear(self.in_dim, self.num_quad_prep, bias=False)  # First transformation for quadratic
        self.quad_linear_prep = nn.Linear(self.num_quad_prep, self.out_dim, bias=False)  # Second transformation for quadratic
        self.quadratic.weight.data = self.gram_schmidt(self.quadratic.weight.data)  # Initialize weights orthogonally
        # Phase 2 parameters
        # TODO: further development: 
        # Decompose quad_linear_prep into 2: quad_linear_prep1 and quad_linear.
        # quad_linear_prep1: Linear(num_quad_prep-num_quad, out_dim)
        # quad_indices[0] <= ~[1]
        self.quad_indices = torch.zeros([num_quad, 2], dtype=torch.int16)
        self.quad_linear = nn.Linear(self.num_quad, self.out_dim, bias=False)
        nn.init.constant_(self.quad_linear.weight, 0)

    def forward(self, x):
        # TODO: switch(self.stage)
        if self.prep_stage:
            y_quadratic = self.quadratic(x)
            out_quadratic = torch.square(y_quadratic)
            out_quadratic = self.quad_linear_prep(out_quadratic)
        else:
            out_quadratic = self.calc_quads(x)
            out_quadratic = self.quad_linear(out_quadratic)

        out_linear = self.linear(x)
        return out_linear + out_quadratic
    
    def calc_quads(self, x):
        # out_quadratic = torch.stack([func(x_T) for func in self.significant_quad_funcs]).T
        # x.shape = [B, C]
        out_quadratic = torch.stack([x[:, self.quad_indices[i][0], ...] * x[:, self.quad_indices[i][1], ...] for i in range(self.num_quad)], dim=1)
        return out_quadratic
    
    def normalize(self):
        """
        Normalizes the rows of the quadratic layer and adjusts the quad_linear_prep accordingly.
        """
        with torch.no_grad():
            norms = torch.norm(self.quadratic.weight.data, dim=1, keepdim=True)
            self.quadratic.weight.data /= norms
            self.quad_linear_prep.weight.data *= norms.squeeze()

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
            _, sorted_indices = torch.sort(self.quad_linear_prep.weight.abs().sum(dim=0), descending=True)
            self.quadratic.weight.data = self.quadratic.weight.data[sorted_indices]
            self.quad_linear_prep.weight.data = self.quad_linear_prep.weight.data[:, sorted_indices]

    def orthonormalize(self):
        self.normalize()
        self.reorder()
        self.gram_schmidt(self.quadratic.weight.data)

    def orthonormalize_prep(self):
        if self.prep_stage:
            self.orthonormalize

    def orthogonality_loss(self):
        """
        A loss function that encourages the weights of the quadratic layer to be orthogonal.
        """
        w = self.quadratic.weight
        gram = torch.mm(w, w.t())
        eye = torch.eye(gram.shape[0], device=gram.device)
        loss = torch.norm(gram - eye)
        return loss

    def get_quad_expr(self, x):
        quadratic_weights = self.quadratic.weight.data.numpy()
        quad_linear_prep_weights = self.quad_linear_prep.weight.data.numpy()
        y_quadratic = sp.Matrix(quadratic_weights) * x
        out_quadratic = sp.Matrix(quad_linear_prep_weights) * y_quadratic.applyfunc(lambda a: a**2) 
        return out_quadratic
    
    def model_to_sympy(self, x):
        linear_weights = self.linear.weight.data.numpy()
        linear_bias = self.linear.bias.data.numpy()
        out_linear = sp.Matrix(linear_weights) * x + sp.Matrix(linear_bias)
        out_quadratic = self.get_quad_expr(x)
        return out_quadratic + out_linear
    
    def get_mat(self, linear):
        return linear.weight.data[0]

    # replace previous representations if the proportion of new representation reaches threshold
    def replace_previous(self, threshold=0.0):
        Q = self.get_mat(self.quadratic)
        Lams = self.get_mat(self.quad_linear_prep)
        print(f"Q.shape:{Q.shape}, Lam1.shape: {Lams.diag_embed().shape}")
        n = Lams.shape[0]
        A = torch.einsum('ab,nbc ->nac', Q.T, Lams.diag_embed())
        A = torch.einsum('nab,bc->nac', A, Q)
        # A = Q.T @ Lams.diag_embed() @ Q
        B = torch.stack([torch.triu(2. * A[i] - torch.diag(A[i].diag())) for i in range(n)])  
        B1 = torch.sum(torch.abs(B), dim=0)
        # aggregate all expressions of quad terms
        # for i in range(self.num_quad):
        #     idx = self.quad_indices[i, :]
        #     B1[idx[0], idx[1]] += self.quad_linear.weight.data[i].item()

        B_view = B1.view(-1)
        indices = torch.topk(abs(B_view), self.num_quad)
        proportion = torch.sum(B_view[indices.indices])/torch.sum(B_view)
            
        if proportion < threshold:
            return proportion

        self.prep_stage = False

        # Update quad_indices and quad_linear
        for i in range(self.num_quad):
            idx = indices[1][i]
            self.quad_indices[i] = get_mat_index(idx, A.shape[1])
            self.quad_linear.weight.data[i] = B_view[idx]

        # Update the initial matrices
        A_new = A.clone()
        for idx in self.quad_indices:
            A_new[idx[0], idx[1]] = 0
            A_new[idx[1], idx[0]] = 0

        L, V = torch.linalg.eig(A_new)

        self.quadratic.weight.data = V.real.clone()[:self.num_quad, :]
        self.quad_linear_prep.weight.data = L[:self.num_quad_prep].real.clone()
        return proportion


class QuadraticLayer2D(QuadraticLayer):
    def __init__(self, in_dim, num_quad_prep, num_quad, out_dim):
        super().__init__(in_dim, num_quad_prep, num_quad, out_dim)
        self.linear = nn.Conv2d(self.in_dim, self.out_dim, 1)
        self.quadratic = nn.Conv2d(self.in_dim, self.num_quad_prep, 1)  # First transformation for quadratic
        self.quad_linear_prep = nn.Conv2d(self.num_quad_prep, self.out_dim, 1)  # Second transformation for quadratic
        self.quad_linear = nn.Conv2d(self.num_quad, self.out_dim, 1)

    def get_mat(self, conv):
        return conv.weight.data.squeeze(-1).squeeze(-1)
    
    def replace_previous(self, threshold=0.0):
        Q = self.get_mat(self.quadratic)
        Lams = self.get_mat(self.quad_linear_prep)
        print(f"Q.shape:{Q.shape}, Lam1.shape: {Lams.diag_embed().shape}")
        n = Lams.shape[0]
        A = torch.einsum('ab,nbc ->nac', Q.T, Lams.diag_embed())
        A = torch.einsum('nab,bc->nac', A, Q)
        # A = Q.T @ Lams.diag_embed() @ Q
        B = torch.stack([torch.triu(2. * A[i] - torch.diag(A[i].diag())) for i in range(n)])  
        B1 = torch.sum(torch.abs(B), dim=0)
        # aggregate all expressions of quad terms
        # for i in range(self.num_quad):
        #     idx = self.quad_indices[i, :]
        #     B1[idx[0], idx[1]] += self.quad_linear.weight.data[i].item()

        B_view = B1.view(-1)
        indices = torch.topk(abs(B_view), self.num_quad)
        proportion = torch.sum(B_view[indices.indices])/torch.sum(B_view)
            
        if proportion < threshold:
            return proportion
        
        self.prep_stage = False
        # Update quad_indices and quad_linear
        for i in range(self.num_quad):
            idx = indices[1][i]
            self.quad_indices[i] = get_mat_index(idx, A.shape[1])
            self.quad_linear.weight.data[i] = B_view[idx]

        # Update the initial matrices
        A_new = A.clone()
        for idx in self.quad_indices:
            A_new[idx[0], idx[1]] = 0
            A_new[idx[1], idx[0]] = 0

        L, V = torch.linalg.eig(A_new)
        print(f"A_new.shape: {A_new.shape}, L.shape: {L.shape}, V.shape: {V.shape}")

        print(f"quadratic.shape: {self.quadratic.weight.data.shape}, quad_linear_prep.shape: {self.quad_linear_prep.weight.data.shape}")
        self.quadratic.weight.data = V.real.clone()[:self.num_quad, :]
        self.quad_linear_prep.weight.data = L[:self.num_quad_prep].real.clone()
        print(f"quadratic.shape: {self.quadratic.weight.data.shape}, quad_linear_prep.shape: {self.quad_linear_prep.weight.data.shape}")
        return proportion


class QuadPath(nn.Module):
    def __init__(self, in_dim, out_dim, num_quad, num_prod):
        super(QuadPath, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_quad = num_quad
        self.num_prod = num_prod
        self.out_range = torch.tensor(np.array(range(out_dim)), dtype=torch.int16)
        
        self.prep_stage = True
        # Stage 1 params
        self.quadratic = nn.Linear(self.in_dim, self.num_quad, bias=False)  # First transformation for quadratic
        self.quadratic.weight.data = self.gram_schmidt(self.quadratic.weight.data)  # Initialize weights orthogonally
        self.quad_linear = nn.Linear(self.num_quad, self.out_dim, bias=False)  # Second transformation for quadratic
        # Stage 2 params
        self.prod_indices = torch.zeros([num_prod, 2], dtype=torch.int16)  # quad_indices[0] <= ~[1]
        self.prod_linear = nn.Linear(num_prod, self.out_dim, bias=False)
        nn.init.xavier_uniform_(self.quad_linear.weight, 1)

    def forward(self, x):
        if self.prep_stage:
            y_quadratic = self.quadratic(x)
            out_quadratic = torch.square(y_quadratic)
            out_quadratic = self.quad_linear(out_quadratic)
        else:
            out_quadratic = self.calc_quads(x)
            out_quadratic = self.prod_linear(out_quadratic)

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
    def __init__(self, in_dim, num_prod, out_dim):
        super(ProductLayer, self).__init__()
        self.in_dim = in_dim
        self.num_prods = num_prod
        self.out_dim = out_dim
        assert in_dim >= 2*num_prod, "Error: in_dim < 2*num_prods!"
        self.in_dim_linear = in_dim + num_prod
        self.linear = nn.Linear(self.in_dim_linear, self.out_dim, bias=False)

    def get_prods(self, x):
        return torch.stack([x[:, 2 *i] * x[:, 2 *i + 1] for i in torch.arange(0, self.num_prods-1, dtype=torch.int8)], dim=1)

    def forward(self, x):
        x = torch.cat((x, self.get_prods(x)), dim=-1)
        x = self.linear(x)
        return x
    
class ProductLayer2D(ProductLayer):
    def __init__(self, in_dim, num_prods, out_dim):
        super().__init__(in_dim, num_prods, out_dim)
        self.linear = nn.Conv2d(self.in_dim_linear, self.out_dim, 1)
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
    def __init__(self, in_dim, num_prod, p=1, clamp_thresh=16.):
        super(ProductGating, self).__init__()
        self.in_dim = in_dim
        self.num_prod = num_prod
        assert in_dim >= 2*num_prod, "Error: in_dim < 2*num_prod!"
        self.in_dim_linear = in_dim + num_prod
        self.range_prod = torch.arange(0, num_prod, 1, dtype=torch.int)
        # TODO: more initialization!
        # gating_coeff.size: [num_prod] (the first dimension for broadcast on batches)
        self.gating_coeff = 0.5 * torch.ones([num_prod])
        self.p = p
        self.clamp_thresh = nn.Parameter(torch.tensor(clamp_thresh))

    def forward(self, x):
        prods = torch.stack([x[:, 2 * i, ...] * x[:, 2 * i + 1, ...] for i in self.range_prod], dim=1)
        prods = torch.clamp(prods, min=-self.clamp_thresh, max=self.clamp_thresh)
        # self.gating_coeff.clamp_(-self.eps, 1.+self.eps)
        coeff = torch.pow(torch.clamp(self.gating_coeff, min=0, max=1), self.p).view(1, -1, *([1] * (x.dim() - 2))).to(x.device)

        tmp = prods * coeff
        tmp = tmp + x[:, -self.num_prod:, ...] * (1. - coeff)
        new_x = x.clone()
        new_x[:, -self.num_prod:, ...] = tmp
        return new_x
