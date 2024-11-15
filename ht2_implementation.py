import torch
import tensorly as tl
from torch import nn
from ht2_decomposition import ht2_decomposition, get_singular_values


def estimate_ranks(weights, energy_threshold):
    ranks = []
    dims = weights.shape

    unfold_0 = tl.unfold(weights, 0)
    unfold_2 = tl.unfold(weights, 2)
    unfold_0_1 = weights.reshape(dims[0] * dims[1], -1)

    for unfold in (unfold_0, unfold_2, unfold_0_1):
        s = get_singular_values(
            unfold,
        )
        total_sum = torch.sum(s**2)
        s_sum = 0
        count = 0

        for i in s:
            s_sum += i**2
            count += 1
            energy = s_sum / total_sum
            if energy > energy_threshold:
                ranks.append(count)
                break

    return ranks


def ht2(layer, energy_threshold):
    tl.set_backend("pytorch")
    is_bias = torch.is_tensor(layer.bias)
    weights = layer.weight.detach()
    weights = weights.moveaxis(0, 2)
    ranks = estimate_ranks(weights, energy_threshold)
    R1, R3, R13 = ranks
    first, fourth, second, third = ht2_decomposition(weights, ranks)

    first_weights = first.T.unsqueeze(2).unsqueeze(3)
    second_weights = second.moveaxis(-1, 0).unsqueeze(3)
    third_weights = third.moveaxis(0, 1).unsqueeze(2)
    fourth_weights = fourth.unsqueeze(2).unsqueeze(3)

    first_layer = nn.Conv2d(
        in_channels=layer.in_channels,
        out_channels=R1,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=False,
    )

    second_layer = nn.Conv2d(
        in_channels=R1,
        out_channels=R13,
        kernel_size=(layer.kernel_size[0], 1),
        stride=(layer.stride[0], 1),
        padding=(layer.padding[0], 0),
        dilation=layer.dilation,
        bias=False,
    )

    third_layer = nn.Conv2d(
        in_channels=R13,
        out_channels=R3,
        kernel_size=(1, layer.kernel_size[1]),
        stride=(1, layer.stride[1]),
        padding=(0, layer.padding[1]),
        dilation=layer.dilation,
        bias=False,
    )

    fourth_layer = nn.Conv2d(
        in_channels=R3,
        out_channels=layer.out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=is_bias,
    )

    first_layer.weight.data = first_weights
    second_layer.weight.data = second_weights
    third_layer.weight.data = third_weights
    fourth_layer.weight.data = fourth_weights

    if is_bias:
        fourth_layer.bias.data = layer.bias.data

    return nn.Sequential(*[first_layer, second_layer, third_layer, fourth_layer])
