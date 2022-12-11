import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .. import params as p


def unflatten_and_split_apart_batches(x: torch.Tensor) -> torch.Tensor:
    """
    Transforms output of decoder which is in format (Batch*H*W,1,Channels) to (Batch,Channels,H,W)
    """
    H = int(math.sqrt(x.shape[0]/p.BATCH_SIZE))
    x = x.reshape((p.BATCH_SIZE, H, H, -1))
    return x.moveaxis(-1, 1)

def flatten_spacial_dims(x: torch.Tensor) -> torch.Tensor:
    """
    Flattens a numpy array from dimensions (batch, channels, height, width) to (batch*height*width, 1, channels)
    """
    if not len(x.shape) == 4:
        raise ValueError(
            """Input is expected in format (Batch, Channels, Height, Width).\n
                Shape was instead: """ + str(x.shape))
    # Move channel dimension in the back
    x=x.moveaxis(1, -1)
    # Then flatten front three dimension together
    x=x.reshape(-1, x.shape[-1])
    # To make conv layers work, we need a dimension of size 1 in the middle
    return x.unsqueeze(1)


def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`
    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find
    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)


def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size=state_dict[state_dict_key].size()
    registered_buf=find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(
                f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(
            new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.
    (There's no way in torch to directly load a buffer with a dynamic size)
    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    valid_buffer_names=[n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",
            state_dict,
            policy,
            dtype,
        )


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )
