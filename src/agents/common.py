import numpy as np
from torch import nn, Tensor
from typing import Callable

def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """
    Initialize a layer with orthogonal weights and constant bias.
    Args:
        layer (nn.Module): The layer to initialize.
        std (float, optional): Standard deviation for weight initialization. Default is sqrt(2).
        bias_const (float, optional): Constant value for bias initialization. Default is 0.0.
    Returns:
        nn.Module: The initialized layer.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class ResidualBlock(nn.Module):
    """
    A residual block with two convolutional layers and a residual connection.
    Args:
        channels (int): Number of input and output channels for the convolutional layers.
        kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
        padding (int, optional): Padding for the convolutional layers. Default is 1.
        Activation (Callable, optional): Activation function to apply after each convolution. Default is ReLU.
    """
    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1, Activation: Callable = nn.functional.relu) -> None:
        super().__init__()
        self._activation = Activation
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x: Tensor) -> Tensor:
        inputs = x
        x = self._activation(x)
        x = self.conv0(x)
        x = self._activation(x)
        x = self.conv1(x)
        return x + inputs # residual connection


class ConvSequence(nn.Module):
    """
    A sequence of convolutional layers with residual blocks and max pooling.
    Args:
        input_shape (tuple[int, int, int]): Input shape in (C, H, W) format.
        out_channels (int): Number of output channels for the first convolutional layer.
        kernel_size (int): Size of the convolutional kernel.
        padding (int): Padding for the convolutional layers.
        stride (int): Stride for the max pooling operation.
        residual_activation (Callable, optional): Activation function for the residual blocks. Default is ReLU.
    """
    def __init__(self, input_shape: tuple[int, int, int], out_channels: int, kernel_size: int, padding: int, stride: int, residual_activation: Callable) -> None:
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._padding = padding
        self._stride = stride
        act = getattr(nn.functional, residual_activation.value.lower() if residual_activation else "relu", None)
        assert act is not None, f"Residual block activation function '{residual_activation}' is not supported. Please use a valid activation function from torch.nn.functional."

        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=self._kernel_size, padding=self._padding)
        self.res_block0 = ResidualBlock(self._out_channels, kernel_size=self._kernel_size, padding=self._padding, Activation=act)
        self.res_block1 = ResidualBlock(self._out_channels, kernel_size=self._kernel_size, padding=self._padding, Activation=act)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=self._kernel_size, stride=self._stride, padding=self._padding)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self) -> tuple[int, int, int]:
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)
