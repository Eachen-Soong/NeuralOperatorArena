from functools import partialmethod
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from ..layers.spectral_convolution import SpectralConv
from ..layers.spherical_convolution import SphericalConv
from ..layers.new_spectral_conv import SpectralConvProd, SpectralConvAttn2d
from ..layers.padding import DomainPadding
from ..layers.fno_block import FNOBlocks1, F_FNOBlocks2D, DimFNOBlocks
from ..layers.mlp import MLP


def get_grid_positional_encoding(input_tensor, grid_boundaries=[[0,1],[0,1]], channel_dim=1):
    """
        Appends grid positional encoding to an input tensor, concatenating as additional dimensions along the channels
    """
    shape = list(input_tensor.shape)
    height, width = shape[-2:]
    
    xt = torch.linspace(grid_boundaries[0][0], grid_boundaries[0][1], height + 1)[:-1]
    yt = torch.linspace(grid_boundaries[1][0], grid_boundaries[1][1], width + 1)[:-1]

    grid_x, grid_y = torch.meshgrid(xt, yt, indexing='ij')

    if len(shape) == 2:
        grid_x = grid_x.repeat(1, 1).unsqueeze(channel_dim)
        grid_y = grid_y.repeat(1, 1).unsqueeze(channel_dim)
    else:
        grid_x = grid_x.repeat(1, 1).unsqueeze(0).unsqueeze(channel_dim)
        grid_y = grid_y.repeat(1, 1).unsqueeze(0).unsqueeze(channel_dim)
    
    return torch.cat([grid_x, grid_y], dim=channel_dim)


def get_grid_positional_encoding_3d(input_tensor, grid_boundaries=[[0,1],[0,1],[0,1]], channel_dim=1):
    """
        Appends grid positional encoding to an input tensor, concatenating as additional dimensions along the channels
    """
    shape = list(input_tensor.shape)
    height, width, depth = shape[-3:]
    
    xt = torch.linspace(grid_boundaries[0][0], grid_boundaries[0][1], height + 1)[:-1]
    yt = torch.linspace(grid_boundaries[1][0], grid_boundaries[1][1], width + 1)[:-1]
    zt = torch.linspace(grid_boundaries[2][0], grid_boundaries[2][1], depth + 1)[:-1]

    grid_x, grid_y, grid_z = torch.meshgrid(xt, yt, zt, indexing='ijk')

    if len(shape) == 3:
        grid_x = grid_x.repeat(1, 1).unsqueeze(channel_dim)
        grid_y = grid_y.repeat(1, 1).unsqueeze(channel_dim)
        grid_z = grid_z.repeat(1, 1).unsqueeze(channel_dim)
    else:
        grid_x = grid_x.repeat(1, 1).unsqueeze(0).unsqueeze(channel_dim)
        grid_y = grid_y.repeat(1, 1).unsqueeze(0).unsqueeze(channel_dim)
        grid_z = grid_z.repeat(1, 1).unsqueeze(0).unsqueeze(channel_dim)

    return grid_x, grid_y, grid_z


class FNO(nn.Module):
    """N-Dimensional Fourier Neural Operator

    Parameters
    ----------
    n_modes : int tuple
        number of modes to keep in Fourier Layer, along each dimension
        The dimensionality of the TFNO is inferred from ``len(n_modes)``
    hidden_channels : int
        width of the FNO (i.e. number of channels)
    in_channels : int, optional
        Number of input channels, by default 3
    out_channels : int, optional
        Number of output channels, by default 1
    lifting_channels : int, optional
        number of hidden channels of the lifting block of the FNO, by default 256
    projection_channels : int, optional
        number of hidden channels of the projection block of the FNO, by default 256
    n_layers : int, optional
        Number of Fourier Layers, by default 4
    incremental_n_modes : None or int tuple, default is None
        * If not None, this allows to incrementally increase the number of
          modes in Fourier domain during training. Has to verify n <= N
          for (n, m) in zip(incremental_n_modes, n_modes).

        * If None, all the n_modes are used.

        This can be updated dynamically during training.
    fno_block_precision : str {'full', 'half', 'mixed'}
        if 'full', the FNO Block runs in full precision
        if 'half', the FFT, contraction, and inverse FFT run in half precision
        if 'mixed', the contraction and inverse FFT run in half precision
    stabilizer : str {'tanh'} or None, optional
        By default None, otherwise tanh is used before FFT in the FNO block
    use_mlp : bool, optional
        Whether to use an MLP layer after each FNO block, by default False
    mlp_dropout : float , optional
        droupout parameter of MLP layer, by default 0
    mlp_expansion : float, optional
        expansion parameter of MLP layer, by default 0.5
    non_linearity : nn.Module, optional
        Non-Linearity module to use, by default F.gelu
    norm : F.module, optional
        Normalization layer to use, by default None
    preactivation : bool, default is False
        if True, use resnet-style preactivation
    fno_skip : {'linear', 'identity', 'soft-gating'}, optional
        Type of skip connection to use in fno, by default 'linear'
    mlp_skip : {'linear', 'identity', 'soft-gating'}, optional
        Type of skip connection to use in mlp, by default 'soft-gating'
    separable : bool, default is False
        if True, use a depthwise separable spectral convolution
    factorization : str or None, {'', 'tucker', 'cp', 'tt'}
        Tensor factorization of the parameters weight to use, by default None.
        * If None, a dense tensor parametrizes the Spectral convolutions
        * Otherwise, the specified tensor factorization is used.
    joint_factorization : bool, optional
        Whether all the Fourier Layers should be parametrized by a single tensor
        (vs one per layer), by default False
    rank : float or rank, optional
        Rank of the tensor factorization of the Fourier weights, by default 1.0
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
    implementation : {'factorized', 'reconstructed'}, optional, default is 'factorized'
        If factorization is not None, forward mode to use:
        * `reconstructed` : the full weight tensor is reconstructed from the
          factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of
          the decomposition
    decomposition_kwargs : dict, optional, default is {}
        Optionaly additional parameters to pass to the tensor decomposition
    domain_padding : None or float, optional
        If not None, percentage of padding to use, by default None
    domain_padding_mode : {'symmetric', 'one-sided'}, optional
        How to perform domain padding, by default 'one-sided'
    fft_norm : str, optional
        by default 'forward'
    positional_encoding : bool, whether to append the positional encoding.
    """

    def __init__(
        self,
        n_modes,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        output_scaling_factor=None,
        incremental_n_modes=None,
        fno_block_precision="full",
        channel_mixing="",
        mlp_dropout=0,
        mlp_expansion=0.5,
        num_prod=2,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        preactivation=False,
        fno_skip="linear",
        mixer_skip="soft-gating",
        separable=False,
        factorization='',
        rank=1.0,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="one-sided",
        fft_norm="forward",
        positional_encoding=False,
        SpectralConv=SpectralConv,
        **kwargs
    ):
        super().__init__()
        self.n_dim = len(n_modes)
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        self.positional_encoding = None
        self.positional_encoding_cached = None
        if positional_encoding:
            self.in_channels += self.n_dim
            if self.n_dim == 2:
                self.positional_encoding = get_grid_positional_encoding
            if self.n_dim == 3:
                self.positional_encoding = get_grid_positional_encoding_3d
        
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.rank = rank
        if factorization == '': factorization = None # newly added
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = (fno_skip,)
        self.mixer_skip = (mixer_skip,)
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.channel_mixing = channel_mixing
        self.separable = separable
        self.preactivation = preactivation
        self.fno_block_precision = fno_block_precision

        # See the class' property for underlying mechanism
        # When updated, change should be reflected in fno blocks
        self._incremental_n_modes = incremental_n_modes

        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode,
                output_scaling_factor=output_scaling_factor,
            )
        else:
            self.domain_padding = None

        self.domain_padding_mode = domain_padding_mode

        if output_scaling_factor is not None and not joint_factorization:
            if isinstance(output_scaling_factor, (float, int)):
                output_scaling_factor = [output_scaling_factor] * self.n_layers
        self.output_scaling_factor = output_scaling_factor

        self.fno_blocks = FNOBlocks1(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            output_scaling_factor=output_scaling_factor,
            channel_mixing=channel_mixing,
            mlp_dropout=mlp_dropout,
            mlp_expansion=mlp_expansion,
            num_prod=num_prod,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            preactivation=preactivation,
            fno_skip=fno_skip,
            mixer_skip=mixer_skip,
            incremental_n_modes=incremental_n_modes,
            fno_block_precision=fno_block_precision,
            rank=rank,
            fft_norm=fft_norm,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
            SpectralConv=SpectralConv,
            n_layers=n_layers,
            **kwargs
        )

        # if lifting_channels is passed, make lifting an MLP
        # with a hidden layer of size lifting_channels
        if self.lifting_channels:
            self.lifting = MLP(
                in_channels=self.in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
            )
        # otherwise, make it a linear layer
        else:
            self.lifting = MLP(
                in_channels=self.in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
            )
        self.projection = MLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )

    def forward(self, x, output_shape=None, **kwargs):
        """TFNO's forward pass

        Parameters
        ----------
        x : tensor
            input tensor
        output_shape : {tuple, tuple list, None}, default is None
            Gives the option of specifying the exact output shape for odd shaped inputs.
            * If None, don't specify an output shape
            * If tuple, specifies the output-shape of the **last** FNO Block
            * If tuple list, specifies the exact output-shape of each FNO Block
        """
        if self.positional_encoding is not None:
            if self.positional_encoding_cached is None:
                self.positional_encoding_cached = self.positional_encoding(x).to(x.device)
            grids = self.positional_encoding_cached.repeat(x.shape[0], *([1]*(self.n_dim+1)))
            x = torch.cat([x, grids], dim=1)
        if output_shape is None:
            output_shape = [None]*self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None]*(self.n_layers - 1) + [output_shape]

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)
        
        x = self.projection(x)

        return x

    @property
    def incremental_n_modes(self):
        return self._incremental_n_modes

    @incremental_n_modes.setter
    def incremental_n_modes(self, incremental_n_modes):
        self.fno_blocks.incremental_n_modes = incremental_n_modes


class FNO1d(FNO):
    """1D Fourier Neural Operator

    For the full list of parameters, see :class:`neuralop.models.FNO`.

    Parameters
    ----------
    modes_height : int
        number of Fourier modes to keep along the height
    """

    def __init__(
        self,
        n_modes_height,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        incremental_n_modes=None,
        fno_block_precision="full",
        n_layers=4,
        output_scaling_factor=None,
        non_linearity=F.gelu,
        stabilizer=None,
        channel_mixing="",
        mlp_dropout=0,
        mlp_expansion=0.5,
        num_prod=2,
        norm=None,
        skip="soft-gating",
        separable=False,
        preactivation=False,
        factorization=None,
        rank=1.0,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="one-sided",
        fft_norm="forward",
        **kwargs
    ):
        super().__init__(
            n_modes=(n_modes_height,),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            output_scaling_factor=None,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            channel_mixing=channel_mixing,
            mlp_dropout=mlp_dropout,
            mlp_expansion=mlp_expansion,
            num_prod=num_prod,
            incremental_n_modes=incremental_n_modes,
            fno_block_precision=fno_block_precision,
            norm=norm,
            skip=skip,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization,
            rank=rank,
            joint_factorization=joint_factorization,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
            fft_norm=fft_norm,
        )
        self.n_modes_height = n_modes_height


class FNO2d(FNO):
    """2D Fourier Neural Operator

    For the full list of parameters, see :class:`neuralop.models.FNO`.

    Parameters
    ----------
    n_modes_width : int
        number of modes to keep in Fourier Layer, along the width
    n_modes_height : int
        number of Fourier modes to keep along the height
    """

    def __init__(
        self,
        n_modes_height,
        n_modes_width,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        output_scaling_factor=None,
        incremental_n_modes=None,
        fno_block_precision="full",
        non_linearity=F.gelu,
        stabilizer=None,
        channel_mixing="",
        mlp_dropout=0,
        mlp_expansion=0.5,
        num_prod=2,
        norm=None,
        skip="soft-gating",
        separable=False,
        preactivation=False,
        factorization=None,
        rank=1.0,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="one-sided",
        fft_norm="forward",
        **kwargs
    ):
        super().__init__(
            n_modes=(n_modes_height, n_modes_width),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            output_scaling_factor=None,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            channel_mixing=channel_mixing,
            mlp_dropout=mlp_dropout,
            mlp_expansion=mlp_expansion,
            num_prod=num_prod,
            incremental_n_modes=incremental_n_modes,
            fno_block_precision=fno_block_precision,
            norm=norm,
            skip=skip,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization,
            rank=rank,
            joint_factorization=joint_factorization,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
            fft_norm=fft_norm,
        )
        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width


class FNO3d(FNO):
    """3D Fourier Neural Operator

    For the full list of parameters, see :class:`neuralop.models.FNO`.

    Parameters
    ----------
    modes_width : int
        number of modes to keep in Fourier Layer, along the width
    modes_height : int
        number of Fourier modes to keep along the height
    modes_depth : int
        number of Fourier modes to keep along the depth
    """

    def __init__(
        self,
        n_modes_height,
        n_modes_width,
        n_modes_depth,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        output_scaling_factor=None,
        incremental_n_modes=None,
        fno_block_precision="full",
        non_linearity=F.gelu,
        stabilizer=None,
        channel_mixing="",
        mlp_dropout=0,
        mlp_expansion=0.5,
        num_prod=2,
        norm=None,
        skip="soft-gating",
        separable=False,
        preactivation=False,
        factorization=None,
        rank=1.0,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="one-sided",
        fft_norm="forward",
        **kwargs
    ):
        super().__init__(
            n_modes=(n_modes_height, n_modes_width, n_modes_depth),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            output_scaling_factor=None,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            incremental_n_modes=incremental_n_modes,
            fno_block_precision=fno_block_precision,
            channel_mixing=channel_mixing,
            mlp_dropout=mlp_dropout,
            mlp_expansion=mlp_expansion,
            num_prod=num_prod,
            norm=norm,
            skip=skip,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization,
            rank=rank,
            joint_factorization=joint_factorization,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
            fft_norm=fft_norm,
        )
        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width
        self.n_modes_depth = n_modes_depth


class F_FNO2D(nn.Module):
    def __init__(
        self,
        n_modes,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        output_scaling_factor=None,
        incremental_n_modes=None,
        fno_block_precision="full",
        channel_mixing="",
        mlp_dropout=0,
        mlp_expansion=0.5,
        num_prod=2,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        preactivation=False,
        fno_skip="linear",
        mixer_skip="soft-gating",
        separable=False,
        factorization=None,
        rank=1.0,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="one-sided",
        fft_norm="forward",
        ffno_channel_mixing='add',
        **kwargs
    ):
        super().__init__()
        self.n_dim = len(n_modes)
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = (fno_skip,)
        self.mixer_skip = (mixer_skip,)
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.channel_mixing = channel_mixing
        self.separable = separable
        self.preactivation = preactivation
        self.fno_block_precision = fno_block_precision

        # See the class' property for underlying mechanism
        # When updated, change should be reflected in fno blocks
        self._incremental_n_modes = incremental_n_modes

        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode,
                output_scaling_factor=output_scaling_factor,
            )
        else:
            self.domain_padding = None

        self.domain_padding_mode = domain_padding_mode

        if output_scaling_factor is not None and not joint_factorization:
            if isinstance(output_scaling_factor, (float, int)):
                output_scaling_factor = [output_scaling_factor] * self.n_layers
        self.output_scaling_factor = output_scaling_factor

        self.fno_blocks = F_FNOBlocks2D(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            output_scaling_factor=output_scaling_factor,
            channel_mixing=channel_mixing,
            mlp_dropout=mlp_dropout,
            mlp_expansion=mlp_expansion,
            num_prod=num_prod,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            preactivation=preactivation,
            fno_skip=fno_skip,
            mixer_skip=mixer_skip,
            incremental_n_modes=incremental_n_modes,
            fno_block_precision=fno_block_precision,
            rank=rank,
            fft_norm=fft_norm,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
            ffno_channel_mixing=ffno_channel_mixing,
            n_layers=n_layers,
            **kwargs
        )

        # if lifting_channels is passed, make lifting an MLP
        # with a hidden layer of size lifting_channels
        if self.lifting_channels:
            self.lifting = MLP(
                in_channels=in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
            )
        # otherwise, make it a linear layer
        else:
            self.lifting = MLP(
                in_channels=in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
            )
        self.projection = MLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )

    def forward(self, x, output_shape=None, **kwargs):
        """TFNO's forward pass

        Parameters
        ----------
        x : tensor
            input tensor
        output_shape : {tuple, tuple list, None}, default is None
            Gives the option of specifying the exact output shape for odd shaped inputs.
            * If None, don't specify an output shape
            * If tuple, specifies the output-shape of the **last** FNO Block
            * If tuple list, specifies the exact output-shape of each FNO Block
        """

        if output_shape is None:
            output_shape = [None]*self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None]*(self.n_layers - 1) + [output_shape]
            
        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)
        
        x = self.projection(x)

        return x

    @property
    def incremental_n_modes(self):
        return self._incremental_n_modes

    @incremental_n_modes.setter
    def incremental_n_modes(self, incremental_n_modes):
        self.fno_blocks.incremental_n_modes = incremental_n_modes


class DimFNO(nn.Module):
    """N-Dimensional Fourier Neural Operator

    """
    def __init__(
        self,
        n_modes,
        hidden_channels,
        in_channels=3,
        in_consts=1,
        append_const=False,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        output_scaling_factor=None,
        incremental_n_modes=None,
        fno_block_precision="full",
        channel_mixing="",
        mlp_dropout=0,
        mlp_expansion=0.5,
        num_prod=2,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        align_prediction_dims=[],
        num_consts=1,
        preactivation=False,
        fno_skip="linear",
        mixer_skip="soft-gating",
        separable=False,
        factorization='',
        rank=1.0,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="one-sided",
        fft_norm="forward",
        SpectralConv=SpectralConv,
        positional_encoding=False,
        **kwargs
    ):
        super().__init__()
        self.n_dim = len(n_modes)
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        self.positional_encoding = None
        self.positional_encoding_cached = None
        if positional_encoding:
            self.in_channels += self.n_dim
            if self.n_dim == 2:
                self.positional_encoding = get_grid_positional_encoding
            if self.n_dim == 3:
                self.positional_encoding = get_grid_positional_encoding_3d
        if not append_const and norm == 'dim_norm':
            self.in_channels -= in_consts
        self.append_const = append_const
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.rank = rank
        if factorization == '': factorization = None # newly added
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = (fno_skip,)
        self.mixer_skip = (mixer_skip,)
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.channel_mixing = channel_mixing
        self.separable = separable
        self.preactivation = preactivation
        self.fno_block_precision = fno_block_precision

        # See the class' property for underlying mechanism
        # When updated, change should be reflected in fno blocks
        self._incremental_n_modes = incremental_n_modes

        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode,
                output_scaling_factor=output_scaling_factor,
            )
        else:
            self.domain_padding = None

        self.domain_padding_mode = domain_padding_mode

        if output_scaling_factor is not None and not joint_factorization:
            if isinstance(output_scaling_factor, (float, int)):
                output_scaling_factor = [output_scaling_factor] * self.n_layers
        self.output_scaling_factor = output_scaling_factor

        self.fno_blocks = DimFNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            output_scaling_factor=output_scaling_factor,
            channel_mixing=channel_mixing,
            mlp_dropout=mlp_dropout,
            mlp_expansion=mlp_expansion,
            num_prod=num_prod,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            num_consts=num_consts,
            preactivation=preactivation,
            fno_skip=fno_skip,
            mixer_skip=mixer_skip,
            incremental_n_modes=incremental_n_modes,
            fno_block_precision=fno_block_precision,
            rank=rank,
            fft_norm=fft_norm,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
            SpectralConv=SpectralConv,
            n_layers=n_layers,
            **kwargs
        )

        # if lifting_channels is passed, make lifting an MLP
        # with a hidden layer of size lifting_channels
        if self.lifting_channels:
            self.lifting = MLP(
                in_channels=self.in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
            )
        # otherwise, make it a linear layer
        else:
            self.lifting = MLP(
                in_channels=self.in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
            )
        self.projection = MLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        self.forward = self.forward_dim if norm == 'dim_norm' else self.forward_original
        self.prediction_dims = align_prediction_dims
        self.align_predictions = bool(len(align_prediction_dims))
        if self.align_predictions:
            assert len(align_prediction_dims) == out_channels, 'Number of alignments should be exactly out_channels! Now have: {}, {}'.format(len(align_prediction_dims), out_channels)

        if self.n_dim == 2: broadcast_function = lambda data, spacial_resolution: repeat(data, 'b c -> b c m n', m=spacial_resolution[0], n=spacial_resolution[1])
        elif self.n_dim == 1: broadcast_function = lambda data, spacial_resolution: repeat(data, 'b c -> b c m', m=spacial_resolution[0])
        elif self.n_dim == 3: broadcast_function = lambda data, spacial_resolution: repeat(data, 'b c -> b c m n p', m=spacial_resolution[0], n=spacial_resolution[1], p=spacial_resolution[2])
        self.broadcast_function = broadcast_function

        self.dim_aligner = lambda mu, std, **kwargs: (mu, std, mu, std)
        self.num_consts = num_consts
        
        group_N = hidden_channels//out_channels
        num_group1 = hidden_channels%out_channels
        if num_group1:
            def extend(consts:torch.Tensor):
                return torch.cat([consts[:, :num_group1].repeat(1, group_N+1), consts[:, num_group1:].repeat(1, group_N)], dim=1)
        else:
            def extend(consts:torch.Tensor):
                return consts.repeat(1, group_N)
        if self.n_dim == 2:
            def transform(weight, bias, x):
                return torch.einsum('bc, bcxy -> bcxy', weight, x, ) + repeat(bias, 'b c -> b c x y', x=x.shape[-2], y=x.shape[-1])
        elif self.n_dim == 1:
            def transform(weight, bias, x):
                return torch.einsum('bc, bcx -> bcx', weight, x,) + repeat(bias, 'b c -> b c x', x=x.shape[-1])
        elif self.n_dim == 3:
            def transform(weight, bias, x):
                return torch.einsum('bc, bcxyz -> bcxyz', weight, x, ) + repeat(bias, 'b c -> b c x y z', x=x.shape[-3], y=x.shape[-2], z=x.shape[-1])
        else:
            transform = lambda weight, bias, x: x

        def align_last(x, std, mu):
            weight = extend(std)
            bias = extend(mu)
            x = nn.functional.instance_norm(x)
            return transform(weight, bias, x)
        
        self.align_last = align_last

    def set_dim_aligner(self, function:Callable):
        self.dim_aligner = function

    def set_dimnorm_coeffs(self, weight_coeff, bias_coeff):
        self.fno_blocks.set_dimnorm_coeffs(weight_coeff, bias_coeff)

    def forward_original(self, x, output_shape=None, **kwargs):
        """TFNO's forward pass

        Parameters
        ----------
        x : tensor
            input tensor
        output_shape : {tuple, tuple list, None}, default is None
            Gives the option of specifying the exact output shape for odd shaped inputs.
            * If None, don't specify an output shape
            * If tuple, specifies the output-shape of the **last** FNO Block
            * If tuple list, specifies the exact output-shape of each FNO Block
        """
        if output_shape is None:
            output_shape = [None]*self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None]*(self.n_layers - 1) + [output_shape]
        
        if self.positional_encoding is not None:
            if self.positional_encoding_cached is None:
                self.positional_encoding_cached = self.positional_encoding(x).to(x.device)
            grids = self.positional_encoding_cached.repeat(x.shape[0], *([1]*(self.n_dim+1)))
            x = torch.cat([x, grids], dim=1)

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)
        
        x = self.projection(x)

        return x

    def forward_dim(self, x, output_shape=None, **kwargs):
        """TFNO's forward pass

        Parameters
        ----------
        x : tensor
            input tensor
        output_shape : {tuple, tuple list, None}, default is None
            Gives the option of specifying the exact output shape for odd shaped inputs.
            * If None, don't specify an output shape
            * If tuple, specifies the output-shape of the **last** FNO Block
            * If tuple list, specifies the exact output-shape of each FNO Block
        """
        if output_shape is None:
            output_shape = [None]*self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None]*(self.n_layers - 1) + [output_shape]

        mu, std, aligned_mu, aligned_std = self.dim_aligner(x, **kwargs)

        if self.positional_encoding is not None:
            if self.positional_encoding_cached is None:
                self.positional_encoding_cached = self.positional_encoding(x).to(x.device)
            grids = self.positional_encoding_cached.repeat(x.shape[0], *([1]*(self.n_dim+1)))
            x = torch.cat([x, grids], dim=1)

        if self.append_const:
            consts = self.broadcast_function(kwargs['consts'], x.shape[2:])
            x = torch.cat([x, consts], dim=1)
        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, aligned_mu, aligned_std, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        if self.align_predictions:
            x = self.align_last(x, std[:, self.prediction_dims], mu[:, self.prediction_dims], )
        
        x = self.projection(x)

        return x


    @property
    def incremental_n_modes(self):
        return self._incremental_n_modes

    @incremental_n_modes.setter
    def incremental_n_modes(self, incremental_n_modes):
        self.fno_blocks.incremental_n_modes = incremental_n_modes



def partialclass(new_name, cls, *args, **kwargs):
    """Create a new class with different default values

    Notes
    -----
    An obvious alternative would be to use functools.partial
    >>> new_class = partial(cls, **kwargs)

    The issue is twofold:
    1. the class doesn't have a name, so one would have to set it explicitly:
    >>> new_class.__name__ = new_name

    2. the new class will be a functools object and one cannot inherit from it.

    Instead, here, we define dynamically a new class, inheriting from the existing one.
    """
    __init__ = partialmethod(cls.__init__, *args, **kwargs)
    new_class = type(
        new_name,
        (cls,),
        {
            "__init__": __init__,
            "__doc__": cls.__doc__,
            "forward": cls.forward,
        },
    )
    return new_class

SpecProdFNO = partialclass("SpecProdFNO", FNO, SpectralConv=SpectralConvProd)
TFNO = partialclass("TFNO", FNO, factorization="Tucker")
TFNO1d = partialclass("TFNO1d", FNO1d, factorization="Tucker")
TFNO2d = partialclass("TFNO2d", FNO2d, factorization="Tucker")
TFNO3d = partialclass("TFNO3d", FNO3d, factorization="Tucker")
ProdTFNO = partialclass("ProdTFNO", FNO, factorization="Tucker", channel_mixing="prod-gating")
ProdTFNO1d = partialclass("ProdTFNO1d", FNO1d, factorization="Tucker", channel_mixing="prod-gating")
ProdTFNO2d = partialclass("ProdTFNO2d", FNO2d, factorization="Tucker", channel_mixing="prod-gating")
ProdTFNO3d = partialclass("ProdTFNO3d", FNO3d, factorization="Tucker", channel_mixing="prod-gating")

SFNO = partialclass("SFNO", FNO, factorization="dense", SpectralConv=SphericalConv)
SFNO.__doc__ = SFNO.__doc__.replace("Fourier", "Spherical Fourier", 1)
SFNO.__doc__ = SFNO.__doc__.replace("FNO", "SFNO")
SFNO.__doc__ = SFNO.__doc__.replace("fno", "sfno")