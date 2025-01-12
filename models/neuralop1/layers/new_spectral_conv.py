from typing import List
from .spectral_convolution import *
from .quad_prod import *
from .mlp import MLP
from einops import rearrange

"""
    2 jobs: introducing convolutions between channels and use F-FNO
"""

class SpectralConvFFNO2d(SpectralConv):
    """ 2D Spectral Conv, but destructed the sum of 2* Spectral Conv 1d (equivalent to F-FNO)
        Here we introduce more options, 
        say, you can directly add the outcomes up like the original F-FNO,
        or you can introduce some product terms
        Here we define a channel mixing function, which would mix the channel of the outputs.
    """
    def __init__(self, in_channels, out_channels, n_modes, n_layers=1, channel_mixing='add', num_prod=2, **kwargs):
        # for super(SpectralConv), n_layers=2 * n_layers since we need 2* Spectral Conv 1d
        super().__init__(in_channels=in_channels, out_channels=out_channels, factorization=None, n_modes=n_modes[0], n_layers=2 * n_layers, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        assert len(n_modes) == 2, f"len(n_modes)=={len(n_modes)}, != 2"
        assert n_modes[0] == n_modes[1], f"Currently only supports the same resolution along axises"
        self.n_mode = n_modes[0]
        self.n_layers = n_layers
        self.num_prod = num_prod
        self.merge_mixer = None
        self.linear = None
        if channel_mixing=='prod':
            self.linear = MLP(in_channels=2*out_channels+num_prod, out_channels=out_channels, n_layers=1, n_dim=2)
            def merge_and_mix(a,b):
                x = torch.cat((a, b), dim=1)
                prods = torch.stack([x[:, i, ...] * x[:, out_channels + i, ...] for i in torch.arange(0, self.num_prod, 1, dtype=torch.int)], dim=1)
                x = torch.cat((x, prods), dim=1)
                x = self.linear(x)
                return x
            self.merge_mixer = merge_and_mix
        elif channel_mixing=='linear':
            self.linear = MLP(in_channels=2*out_channels, out_channels=out_channels, n_layers=1, n_dim=2)
            def merge_and_mix(a,b):
                x = torch.cat((a, b), dim=1)
                x = self.linear(x)
                return x
            self.merge_mixer = merge_and_mix
        elif channel_mixing=='add':
            self.merge_mixer = torch.add
        else:
            assert False, 'Unsupported Channel Mixing!'


    def forward(self, x: torch.Tensor, indices=0, **kwargs):
        B, I, M, N = x.shape

        # # # Dimesion Y # # #
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        slices = (
            slice(None),  # Equivalent to: [:,
            slice(None),  # ............... :,
            slice(None),  # ............... :
            slice(self.half_n_modes[0]),  # :half_n_modes[0]]
        )
        if x_fty.dtype == torch.complex32:
            # if x is half precision, run a specialized einsum
            out_ft[slices] = einsum_complexhalf(
                    "bixy,ioy->boxy",
                    x_fty[slices], self._get_weight(2 * indices).to_tensor()
                )
        else:
            out_ft[slices] = tl.einsum(
                    "bixy,ioy->boxy",
                    x_fty[slices], self._get_weight(2 * indices).to_tensor()
                )
                
        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        slices = (
            slice(None),  # Equivalent to: [:,
            slice(None),  # ............... :,
            slice(self.half_n_modes[0]),  # :half_n_modes[0]]
            slice(None),  # ............... :
        )
        if x_ftx.dtype == torch.complex32:
            # if x is half precision, run a specialized einsum
            out_ft[slices] = einsum_complexhalf(
                    "bixy,iox->boxy",
                    x_ftx[slices], self._get_weight(2 * indices + 1).to_tensor()
                )
        else:
            out_ft[slices] = tl.einsum(
                    "bixy,iox->boxy",
                    x_ftx[slices], self._get_weight(2 * indices + 1).to_tensor()
                )

        xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')

        # # # merge and Channel mixing # # #
        x = self.merge_mixer(xx, xy)

        return x


class SpectralConvProd(SpectralConv):
    """ 2D Spectral Conv, but destructed the sum of 2* Spectral Conv 1d (equivalent to F-FNO)
        Here we introduce more options, 
        say, you can directly add the outcomes up like the original F-FNO,
        or you can introduce some product terms
        Here we define a channel mixing function, which would mix the channel of the outputs.
    """
    def __init__(self, in_channels, out_channels, n_modes, n_prods=2, incremental_n_modes=None, bias=True, n_layers=1, separable=False, output_scaling_factor: int | float | List[int | float] | None = None, fno_block_precision="full", rank=0.5, factorization=None, implementation="reconstructed", fixed_rank_modes=False, joint_factorization=False, decomposition_kwargs: dict | None = None, init_std="auto", fft_norm="backward", device=None, dtype=None):
        super().__init__(in_channels, out_channels, n_modes, incremental_n_modes, bias, n_layers, separable, output_scaling_factor, fno_block_precision, rank, factorization, implementation, fixed_rank_modes, joint_factorization, decomposition_kwargs, init_std, fft_norm, device, dtype)
        self.out_channels = out_channels
        self.n_prods = n_prods
        self.prod_path = ProductPath(in_channels, self.n_prods)
        half_total_n_modes = self.half_n_modes
        tensor_kwargs = decomposition_kwargs if decomposition_kwargs is not None else {}
        del self.weight

        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels))**0.5
        else:
            init_std = init_std

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                # If bool, keep the number of layers fixed
                fixed_rank_modes = [0]
            else:
                fixed_rank_modes = None

        # Make sure we are using a Complex Factorized Tensor to parametrize the conv
        if factorization is None:
            factorization = "Dense"  # No factorization
        if not factorization.lower().startswith("complex"):
            factorization = f"Complex{factorization}"

        if separable:
            if in_channels != out_channels:
                raise ValueError(
                    "To use separable Fourier Conv, in_channels must be equal "
                    f"to out_channels, but got in_channels={in_channels} and "
                    f"out_channels={out_channels}",
                )
            weight_shape = (in_channels, *half_total_n_modes)
        else:
            weight_shape = (in_channels, out_channels-self.n_prods, *half_total_n_modes)
        self.separable = separable

        self.n_weights_per_layer = 2 ** (self.order - 1)
        tensor_kwargs = decomposition_kwargs if decomposition_kwargs is not None else {}
        if joint_factorization:
            self.weight = FactorizedTensor.new(
                (self.n_weights_per_layer * n_layers, *weight_shape),
                rank=self.rank,
                factorization=factorization,
                fixed_rank_modes=fixed_rank_modes,
                **tensor_kwargs,
            )
            self.weight.normal_(0, init_std)
        else:
            self.weight = nn.ModuleList(
                [
                    FactorizedTensor.new(
                        weight_shape,
                        rank=self.rank,
                        factorization=factorization,
                        fixed_rank_modes=fixed_rank_modes,
                        **tensor_kwargs,
                    )
                    for _ in range(self.n_weights_per_layer * n_layers)
                ]
            )
            for w in self.weight:
                w.normal_(0, init_std)

    def forward(
        self, x: torch.Tensor, indices=0, output_shape: Optional[Tuple[int]] = None
    ):
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)
        indices : int, default is 0
            if joint_factorization, index of the layers for n_layers > 1

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient

        # Compute Fourier coeffcients
        fft_dims = list(range(-self.order, 0))

        if self.fno_block_precision == "half":
            x = x.half()

        x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)

        if self.fno_block_precision == "mixed":
            # if 'mixed', the above fft runs in full precision, but the
            # following operations run at half precision
            x = x.chalf()

        if self.fno_block_precision in ["half", "mixed"]:
            out_fft = torch.zeros(
                [batchsize, self.out_channels-self.n_prods, *fft_size],
                device=x.device,
                dtype=torch.chalf,
            )
        else:
            out_fft = torch.zeros(
                [batchsize, self.out_channels-self.n_prods, *fft_size],
                device=x.device,
                dtype=torch.cfloat,
            )

        # We contract all corners of the Fourier coefs
        # Except for the last mode: there, we take all coefs as redundant modes
        # were already removed
        mode_indexing = [((None, m), (-m, None)) for m in self.half_n_modes[:-1]] + [
            ((None, self.half_n_modes[-1]),)
        ]

        for i, boundaries in enumerate(itertools.product(*mode_indexing)):
            # Keep all modes for first 2 modes (batch-size and channels)
            idx_tuple = [slice(None), slice(None)] + [slice(*b) for b in boundaries]

            # For 2D: [:, :, :height, :width] and [:, :, -height:, width]
            out_fft[idx_tuple] = self._contract(
                x[idx_tuple],
                self._get_weight(self.n_weights_per_layer * indices + i),
                separable=self.separable,
            )

        if self.output_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple(
                [
                    round(s * r)
                    for (s, r) in zip(mode_sizes, self.output_scaling_factor[indices])
                ]
            )

        if output_shape is not None:
            mode_sizes = output_shape

        out_fft_prod = self.prod_path(out_fft)
        out_fft = torch.cat((out_fft, out_fft_prod), dim=1)

        x = torch.fft.irfftn(out_fft, s=mode_sizes, norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x


class SpectralAttetionBlock2D(nn.Module):
    """
        Attention block as the kernel of 2D spectral convolution.
        Structure:
        partial_spetral shape: B C Rs1 R, where Rs is the spetral resolution
            we would do the attention on the channel dimension, 
            and deem the specreal dimension as the 'series length' dimension.
        full_spetral shape: B C Rs1 Rs2
        Wq shape: H C dk  =einsum(partial_spetral)=> Q shape: B H Rs1 dk R
        Wk shape: H Rs2 C dk =einsum(full_spetral)=> K shape: B H Rs1 dk
        Wv shape: H Rs2 C dv =einsum(full_spetral)=> V shape: B H Rs1 dv (dv = out_channels)
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        dk=0,
        n_heads=1, 
        rank=0.5,
        factorization=None,
        fixed_rank_modes=False,
        decomposition_kwargs: Optional[dict] = None,
        init_std="auto",
        Wv=None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if dk==0:
            dk=in_channels
        self.dk = dk
        self.n_heads = n_heads
        self.dv = int(out_channels/n_heads)
        assert self.dv *n_heads == out_channels, "n_heads must be a divisor of out_channels!"

        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self.n_modes = n_modes
        self.half_n_modes = [m // 2 for m in self.n_modes]
        self.order = len(n_modes)
        assert self.order==2, "Cunrrently only 2D input supported!"

        self.rank = rank
        self.factorization = factorization

        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels))**0.5
        else:
            init_std = init_std

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                # If bool, keep the number of layers fixed
                fixed_rank_modes = [0]
            else:
                fixed_rank_modes = None

        # Make sure we are using a Complex Factorized Tensor to parametrize the
        if factorization is None:
            factorization = "Dense"  # No factorization
        if not factorization.lower().startswith("complex"):
            factorization = f"Complex{factorization}"

        tensor_kwargs = decomposition_kwargs if decomposition_kwargs is not None else {}

        wq_shape = [n_heads, self.in_channels, self.dk]
        self.Wq = FactorizedTensor.new(
                    wq_shape,
                    rank=self.rank,
                    factorization=factorization,
                    fixed_rank_modes=fixed_rank_modes,
                    **tensor_kwargs,
                )
        wk_shape = [n_heads, self.half_n_modes[1], self.in_channels, self.dk]
        self.Wk = FactorizedTensor.new(
                    wk_shape,
                    rank=self.rank,
                    factorization=factorization,
                    fixed_rank_modes=fixed_rank_modes,
                    **tensor_kwargs,
                )
        wv_shape = [n_heads, self.half_n_modes[1], self.in_channels, self.dv]
        self.Wv = FactorizedTensor.new(
                    wv_shape,
                    rank=self.rank,
                    factorization=factorization,
                    fixed_rank_modes=fixed_rank_modes,
                    **tensor_kwargs,
                )

        self.Wq.normal_(0, init_std)
        self.Wk.normal_(0, init_std)
        self.Wv.normal_(0, init_std)

    def complex_softmax(self, z):
        mag = z.abs()
        softmax_mag = torch.nn.functional.softmax(mag, dim=0)
        softmax_real = softmax_mag * z.real / mag
        softmax_imag = softmax_mag * z.imag / mag
        softmax_z = torch.complex(softmax_real, softmax_imag)
        return softmax_z
    
    def forward(self, partial_spetral, full_spetral):
        # partial_spetral: b c s1 r; full_spetral: b c s1 s2
        B, C, S1, R= partial_spetral.shape
        if partial_spetral.dtype == torch.complex32:
            einsum = einsum_complexhalf
        else:
            einsum = tl.einsum
        q = einsum('bcsr,hck->bhskr', partial_spetral, self.Wq)
        k = einsum('bcst,htck->bhsk', full_spetral, self.Wk)
        v = einsum('bcst,htcv->bhsv', full_spetral, self.Wv)
        attn = self.complex_softmax(einsum("bhskr,bhpk->bhspr", q, k))
        out = einsum("bhspr,bhsv->bhvpr", attn, v).view(B, self.out_channels, S1, R)
        return out

# class SpectralAttetionBlock2D(nn.Module):
#     """
#         Attention block as the kernel of 2D spectral convolution.
#         Structure:
#         partial_spetral shape: B C Rs1 R, where Rs is the spetral resolution
#             we would do the attention on the channel dimension, 
#             and deem the specreal dimension as the 'series length' dimension.
#         full_spetral shape: B C Rs1 Rs2
#         Wq shape: H Rs1 dk  =einsum(partial_spetral)=> Q shape: B H C dk R
#         Wk shape: H Rs1 Rs2 dk =einsum(full_spetral)=> K shape: B H C dk
#         Wv shape: H Rs1 Rs2 dv =einsum(full_spetral)=> V shape: B H C dv (dv = Rs1)
#     """
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         n_modes,
#         dk=0,
#         n_heads=1, 
#         rank=1.,
#         factorization=None,
#         fixed_rank_modes=False,
#         decomposition_kwargs: Optional[dict] = None,
#         init_std="auto",
#         Wv=None
#     ):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         if dk==0:
#             dk=in_channels
#         self.dk = dk
#         self.n_heads = n_heads

#         if isinstance(n_modes, int):
#             n_modes = [n_modes]
#         self.n_modes = n_modes
#         self.half_n_modes = [m // 2 for m in self.n_modes]
#         self.order = len(n_modes)
#         assert self.order==2, "Cunrrently only 2D input supported!"

#         self.dv = int(self.half_n_modes[0]/n_heads)
#         assert self.dv *n_heads == self.half_n_modes[0], "n_heads must be a divisor of out_channels!"

#         self.rank = rank
#         self.factorization = factorization

#         if init_std == "auto":
#             init_std = (2 / (in_channels + out_channels))**0.5
#         else:
#             init_std = init_std

#         if isinstance(fixed_rank_modes, bool):
#             if fixed_rank_modes:
#                 # If bool, keep the number of layers fixed
#                 fixed_rank_modes = [0]
#             else:
#                 fixed_rank_modes = None

#         # Make sure we are using a Complex Factorized Tensor to parametrize the
#         if factorization is None:
#             factorization = "Dense"  # No factorization
#         if not factorization.lower().startswith("complex"):
#             factorization = f"Complex{factorization}"

#         tensor_kwargs = decomposition_kwargs if decomposition_kwargs is not None else {}

#         wq_shape = [n_heads, self.half_n_modes[0], self.dk]
#         self.Wq = FactorizedTensor.new(
#                     wq_shape,
#                     rank=self.rank,
#                     factorization=factorization,
#                     fixed_rank_modes=fixed_rank_modes,
#                     **tensor_kwargs,
#                 )
#         wk_shape = [n_heads, self.half_n_modes[0], self.half_n_modes[1], self.dk]
#         self.Wk = FactorizedTensor.new(
#                     wk_shape,
#                     rank=self.rank,
#                     factorization=factorization,
#                     fixed_rank_modes=fixed_rank_modes,
#                     **tensor_kwargs,
#                 )
#         self.Wq.normal_(0, init_std)
#         self.Wk.normal_(0, init_std)

#         wv_shape = [n_heads, self.half_n_modes[0], self.half_n_modes[1], self.dv]
#         if Wv == None:
#             self.Wv = FactorizedTensor.new(
#                         wv_shape,
#                         rank=self.rank,
#                         factorization=factorization,
#                         fixed_rank_modes=fixed_rank_modes,
#                         **tensor_kwargs,
#                     )
#             self.Wv.normal_(0, init_std)
#         else:
#             assert list(Wv.shape) == [n_modes[0], n_modes[1], self.dv], f"Wv.shape should be {[n_modes[0], n_modes[1], self.dv]}, found {Wv.shape}!"
#             self.Wv = Wv

#     def complex_softmax(self, z):
#         mag = z.abs()
#         softmax_mag = torch.nn.functional.softmax(mag, dim=0)
#         softmax_real = softmax_mag * z.real / mag
#         softmax_imag = softmax_mag * z.imag / mag
#         softmax_z = torch.complex(softmax_real, softmax_imag)
#         return softmax_z
    
#     def forward(self, partial_spetral, full_spetral):
#         # partial_spetral: b c s1 r; full_spetral: b c s1 s2
#         B, C, S1, R= partial_spetral.shape
#         if partial_spetral.dtype == torch.complex32:
#             einsum = einsum_complexhalf
#         else:
#             einsum = tl.einsum
#         q = einsum('bcsr,hsk->bhckr', partial_spetral, self.Wq)
#         k = einsum('bcst,hstk->bhck', full_spetral, self.Wk)
#         v = einsum('bcst,hstv->bhcv', full_spetral, self.Wv)
#         attn = self.complex_softmax(einsum("bhckr,bhdk->bhcdr", q, k))
#         out = einsum("bhcdr,bhcv->bhvdr", attn, v).view(B, self.out_channels, S1, R)
#         return out

class SpectralConvAttn2d(BaseSpectralConv):
    """
        2D spectral convolution with an attention block as the kernel.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        dk=0,
        n_heads=2,
        n_layers=1,
        output_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        fno_block_precision="full",
        rank=0.5,
        factorization=None,
        fixed_rank_modes=False,
        decomposition_kwargs: Optional[dict] = None,
        init_std="auto",
        fft_norm="backward",
        device=None,
        dtype=None,
        share_Wv_weight=False,
        **kwargs
    ):
        super().__init__(dtype=dtype, device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        if dk==0:
            dk = in_channels
        self.dk = dk
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.skip = 'mlp'
        
        self.mlp_skip = (self.skip == 'mlp')

        # We index quadrands only
        # n_modes is the total number of modes kept along each dimension
        # half_n_modes is half of that except in the last mode, correponding to
        # the number of modes to keep in *each* quadrant for each dim
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self.n_modes = n_modes
        self.order = len(n_modes)
        assert len(n_modes)==2, "Cunrrently only 2D input supported!"

        self.half_n_modes = [m // 2 for m in self.n_modes]

        # We use half_total_n_modes to build the full weights
        # During training we can adjust incremental_n_modes which will also
        # update half_n_modes
        # So that we can train on a smaller part of the Fourier modes and total
        # weights

        self.fno_block_precision = fno_block_precision
        self.rank = rank
        self.factorization = factorization
        self.output_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(output_scaling_factor, self.order, n_layers)

        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels))**0.5
        else:
            init_std = init_std

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                # If bool, keep the number of layers fixed
                fixed_rank_modes = [0]
            else:
                fixed_rank_modes = None
        self.fft_norm = fft_norm

        if factorization is None:
            factorization = "Dense"  # No factorization
        if not factorization.lower().startswith("complex"):
            factorization = f"Complex{factorization}"
        
        self.share_Wv_weight=share_Wv_weight
        if self.share_Wv_weight:
            tensor_kwargs = decomposition_kwargs if decomposition_kwargs is not None else {}
            self.dv = int(self.half_n_modes[0]/n_heads)
            assert self.dv *n_heads == self.half_n_modes[0], "n_heads must be a divisor of out_channels!"
            wv_shape = [n_heads, self.half_n_modes[0], self.half_n_modes[1], self.dv]
            self.Wvs = nn.ModuleList(
                [
                    FactorizedTensor.new(
                        wv_shape,
                        rank=self.rank,
                        factorization=factorization,
                        fixed_rank_modes=fixed_rank_modes,
                        **tensor_kwargs,
                    )
                    for _ in range(n_layers)
                ]
            )
            for i in range(n_layers):
                self.Wvs[i].normal_(0, init_std)

        else: self.Wvs=[None for _ in range(n_layers)]


        self.attn_blocks_x = nn.ModuleList(
            [
                SpectralAttetionBlock2D(in_channels=in_channels,
                                        out_channels=out_channels,
                                        n_modes=n_modes,
                                        dk=dk,
                                        n_heads=n_heads,
                                        rank=rank,
                                        factorization=factorization,
                                        fixed_rank_modes=fixed_rank_modes,
                                        decomposition_kwargs=decomposition_kwargs,
                                        init_std=init_std,
                                        Wv=self.Wvs[i]
                                        )
                for i in range(n_layers)
            ]
        )
        
        n_modes_prime = [n_modes[1], n_modes[0]]
        self.attn_blocks_y = nn.ModuleList(
            [
                SpectralAttetionBlock2D(in_channels=in_channels,
                                        out_channels=out_channels,
                                        n_modes=n_modes_prime,
                                        dk=dk,
                                        n_heads=n_heads,
                                        rank=rank,
                                        factorization=factorization,
                                        fixed_rank_modes=fixed_rank_modes,
                                        decomposition_kwargs=decomposition_kwargs,
                                        init_std=init_std,
                                        Wv=self.Wvs[i]
                                        )
                for i in range(n_layers)
            ]
        )

        if self.mlp_skip:
            weight_shape = (in_channels, out_channels, *self.half_n_modes)
            self.weight = nn.ModuleList(
                    [
                        FactorizedTensor.new(
                            weight_shape[:2]+weight_shape[2+i%2:3+i%2],
                            rank=self.rank,
                            factorization=factorization,
                            fixed_rank_modes=fixed_rank_modes,
                        )
                        for i in range(2 * n_layers)
                    ]
                )
            for w in self.weight:
                w.normal_(0, init_std)
    
    @property
    def incremental_n_modes(self):
        return self._incremental_n_modes

    @incremental_n_modes.setter
    def incremental_n_modes(self, incremental_n_modes):
        if incremental_n_modes is None:
            self._incremental_n_modes = None
            self.half_n_modes = [m // 2 for m in self.n_modes]

        else:
            if isinstance(incremental_n_modes, int):
                self._incremental_n_modes = [incremental_n_modes] * len(self.n_modes)
            else:
                if len(incremental_n_modes) == len(self.n_modes):
                    self._incremental_n_modes = incremental_n_modes
                else:
                    raise ValueError(
                        f"Provided {incremental_n_modes} for actual "
                        f"n_modes={self.n_modes}."
                    )
            self.weight_slices = [slice(None)] * 2 + [
                slice(None, n // 2) for n in self._incremental_n_modes
            ]
            self.half_n_modes = [m // 2 for m in self._incremental_n_modes]

    def transform(self, x, layer_index=0, output_shape=None):
        in_shape = list(x.shape[2:])

        if self.output_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [
                    round(s * r)
                    for (s, r) in zip(in_shape, self.output_scaling_factor[layer_index])
                ]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(
                x,
                1.0,
                list(range(2, x.ndim)),
                output_shape=out_shape,
            )

    def forward(self, x: torch.Tensor, indices=0, **kwargs):
        B, I, M, N = x.shape

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        x_ft = torch.fft.rfft2(x, norm='ortho')
        # x_ftx.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        if x_ftx.dtype == torch.complex32:
            einsum = einsum_complexhalf
        else:
            einsum = tl.einsum

        slices = (
            slice(None),  # Equivalent to: [:,
            slice(None),  # ............... :,
            slice(self.half_n_modes[0]),  # :half_n_modes[0]]
            slice(None),  # ............... :
        )
        x_ft_slices = (
            slice(None),  # Equivalent to: [:,
            slice(None),  # ............... :,
            slice(self.half_n_modes[0]),  # :half_n_modes[0],
            slice(self.half_n_modes[1]),  # :half_n_modes[1] :
        )

        out_ft[slices] = self.attn_blocks_x[indices](x_ftx[slices], x_ft[x_ft_slices])
        if self.mlp_skip:
            out_ft[slices] += einsum(
                        "bixy,iox->boxy",
                        x_ftx[slices], self.weight[2 * indices].to_tensor()
                    )

        xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')

        # # # Dimesion Y # # #
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        x_ft = x_ft.permute(0, 1, 3, 2)
        # x_fty.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_fty.new_zeros(B, I, N // 2 + 1, M)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]
        slices1 = (
            slice(None),  # Equivalent to: [:,
            slice(None),  # ............... :,
            slice(self.half_n_modes[1]),# :half_n_modes[1],
            slice(None),  # ............... :,]
        )
        slices = (
            slice(None),  # Equivalent to: [:,
            slice(None),  # ............... :,
            slice(None),  # ............... :,
            slice(self.half_n_modes[1]),# :half_n_modes[1],]
        )
        x_ft_slices = (
            slice(None),  # Equivalent to: [:,
            slice(None),  # ............... :,
            slice(self.half_n_modes[1]),  # :half_n_modes[1],
            slice(self.half_n_modes[0]),  # :half_n_modes[0]]
        )
        out_ft[slices1] = self.attn_blocks_y[indices](x_fty[slices].permute(0, 1, 3, 2), x_ft[x_ft_slices])
        out_ft = out_ft.permute(0, 1, 3, 2)
        
        if self.mlp_skip:
            out_ft[slices] += einsum(
                        "bixy,ioy->boxy",
                        x_fty[slices], self.weight[2 * indices].to_tensor()
                    )
        
        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # merge and Channel mixing # # #
        x = xx + xy

        return x
    
    def get_conv(self, indices):
        """Returns a sub-convolutional layer from the joint parametrize main-convolution

        The parametrization of sub-convolutional layers is shared with the main one.
        """
        if self.n_layers == 1:
            Warning("A single convolution is parametrized, directly use the main class.")
            # raise ValueError(
            #     "A single convolution is parametrized, directly use the main class."
            # )

        return SubConv(self, indices)

    def __getitem__(self, indices):
        return self.get_conv(indices)


# class SpectralConvAttn2d(SpectralConv):
#     """ 2D Spectral Conv, but destructed the sum of 2* Spectral Conv 1d (equivalent to F-FNO)
#         Here we introduce more options, 
#         say, you can directly add the outcomes up like the original F-FNO,
#         or you can introduce some product terms
#         Here we define a channel mixing function, which would mix the channel of the outputs.
#     """
#     def __init__(self, in_channels, out_channels, n_modes, n_layers=1, channel_mixing='add', num_prod=2, **kwargs):
#         # for super(SpectralConv), n_layers=2 * n_layers since we need 2* Spectral Conv 1d
#         super().__init__(in_channels=in_channels, out_channels=out_channels, factorization=None, n_modes=n_modes[0], n_layers=2 * n_layers, **kwargs)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.n_modes = n_modes
#         assert len(n_modes) == 2, f"len(n_modes)=={len(n_modes)}, != 2"
#         assert n_modes[0] == n_modes[1], f"Currently only supports the same resolution along axises"
#         self.n_mode = n_modes[0]
#         self.n_layers = n_layers
#         self.num_prod = num_prod
#         self.merge_mixer = None
#         self.linear = None
#         if channel_mixing=='prod':
#             self.linear = MLP(in_channels=2*out_channels+num_prod, out_channels=out_channels, n_layers=1, n_dim=2)
#             def merge_and_mix(a,b):
#                 x = torch.cat((a, b), dim=1)
#                 prods = torch.stack([x[:, i, ...] * x[:, out_channels + i, ...] for i in torch.arange(0, self.num_prod, 1, dtype=torch.int)], dim=1)
#                 x = torch.cat((x, prods), dim=1)
#                 x = self.linear(x)
#                 return x
#             self.merge_mixer = merge_and_mix
#         elif channel_mixing=='linear':
#             self.linear = MLP(in_channels=2*out_channels, out_channels=out_channels, n_layers=1, n_dim=2)
#             def merge_and_mix(a,b):
#                 x = torch.cat((a, b), dim=1)
#                 x = self.linear(x)
#                 return x
#             self.merge_mixer = merge_and_mix
#         elif channel_mixing=='add':
#             self.merge_mixer = torch.add
#         else:
#             assert False, 'Unsupported Channel Mixing!'


#     def forward(self, x: torch.Tensor, indices=0, **kwargs):
#         B, I, M, N = x.shape

#         # # # Dimesion Y # # #
#         x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
#         # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

#         out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1)
#         # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

#         slices = (
#             slice(None),  # Equivalent to: [:,
#             slice(None),  # ............... :,
#             slice(None),  # ............... :
#             slice(self.half_n_modes[0]),  # :half_n_modes[0]]
#         )
#         if x_fty.dtype == torch.complex32:
#             # if x is half precision, run a specialized einsum
#             out_ft[slices] = einsum_complexhalf(
#                     "bixy,ioy->boxy",
#                     x_fty[slices], self._get_weight(2 * indices).to_tensor()
#                 )
#         else:
#             out_ft[slices] = tl.einsum(
#                     "bixy,ioy->boxy",
#                     x_fty[slices], self._get_weight(2 * indices).to_tensor()
#                 )
                
#         xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
#         # x.shape == [batch_size, in_dim, grid_size, grid_size]

#         # # # Dimesion X # # #
#         x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
#         # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

#         out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)
#         # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

#         slices = (
#             slice(None),  # Equivalent to: [:,
#             slice(None),  # ............... :,
#             slice(self.half_n_modes[0]),  # :half_n_modes[0]]
#             slice(None),  # ............... :
#         )
#         if x_ftx.dtype == torch.complex32:
#             # if x is half precision, run a specialized einsum
#             out_ft[slices] = einsum_complexhalf(
#                     "bixy,iox->boxy",
#                     x_ftx[slices], self._get_weight(2 * indices + 1).to_tensor()
#                 )
#         else:
#             out_ft[slices] = tl.einsum(
#                     "bixy,iox->boxy",
#                     x_ftx[slices], self._get_weight(2 * indices + 1).to_tensor()
#                 )

#         xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')

#         # # # merge and Channel mixing # # #
#         x = self.merge_mixer(xx, xy)

#         return x