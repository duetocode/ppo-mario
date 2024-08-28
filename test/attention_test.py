import torch
from ppo_mario.networks.attention import Attention
from ppo_mario.networks.resent import AttentionResBlock


def test_attention_layer():
    """Test the attention layer."""
    # create a random input tensor
    x = torch.randn(3, 32, 64, 64)
    # create the attention layer
    attention = Attention(n_channels=32)
    # pass the input tensor through the attention layer
    y, scale = attention(x)
    # check the output shape
    assert y.shape == x.shape
    # check the output dtype
    assert y.dtype == x.dtype
    # the channel number of the scale should be 1
    assert scale.shape == (3, 1, 64, 64)


def test_attention_resnet_block():
    """Test the attention residual block."""
    # create a random input tensor
    x = torch.randn(3, 32, 64, 64)
    # create the attention residual block
    res_block = AttentionResBlock(32, 64, stride=2)
    # pass the input tensor through the attention residual block
    y = res_block(x)
    # check the output shape
    assert y.shape == (3, 64, 32, 32)
    # check the output dtype
    assert y.dtype == x.dtype
    # the attention map should be preserved
    assert hasattr(res_block, "attention_data")
    assert res_block.attention_data.shape == (3, 1, 32, 32)
