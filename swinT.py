# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 00:28:16 2022

@author: Naive
"""

import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """
    DropPath class
    原理 ：字如其名，Drop Path就是随机将深度学习网络中的多分支结构随机删除。
    功能 ：一般可以作为正则化手段加入网络，但是会增加网络训练的难度。尤其是在神经网络架构搜索NAS问题中，如果设置的drop prob过高，模型甚至有可能不收敛。
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob  # drop_path的比例

    def drop_path(self, inputs):  # inputs:任意形状的Tensor
        # 如果比例为0或者不是训练模式，直接返回原始输入
        if self.drop_prob == 0. or not self.training:
            return inputs
        keep_prob = 1 - self.drop_prob
        keep_prob = torch.to_tensor(keep_prob, dtype='float32')
        shape = (inputs.shape[0], ) + (1, ) * (inputs.ndim - 1
                                               )  # shape=(N, 1, 1, 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=inputs.dtype)
        random_tensor = random_tensor.floor()  # mask
        output = inputs.divide(keep_prob) * random_tensor  # divide是保持相同的输出预期
        return output

    def forward(self, inputs):
        return self.drop_path(inputs)


class Identity(nn.Module):
    """
    Identity layer
    输出和输入完全一致
    可以在一些带有条件语句判断的前向传播层中使用
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class PatchMerging(nn.Module):
    """ Patch Merging class
    将多个patch合并到一起。具体来说，将相邻的2x2patch（dim=C）合并为一个patch。尺寸为4*C的维度被重新缩放为2*C。
    对信息进行了一个压缩，CxHxW-->2CxH/2xW/2。
    Attributes:
        输入分辨率: 有整数组成的二元组
        维度: 单个patch的维度
        降采样: 线性层，将4C映射到2C
        标准化: 在线性层之后进行层标准化
    """
    def __init__(self, input_resolution, dim, out_channels):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim

        self.reduction = nn.Linear(
            4 * dim,
            out_channels,
        )

        self.norm = nn.LayerNorm(4 * dim, )

    def forward(self, x):
        h, w = self.input_resolution
        b, _, c = x.shape
        x = x.reshape([b, h, w, c])

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.reshape([b, -1, 4 * c])  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


class Mlp(nn.Module):
    """ MLP module
    使用线性层实现，激活函数GELU，使用了dropout
    流程: fc -> act -> dropout -> fc -> dropout
    由于残差连接，MLP输出和输入维度保持一致，实际上就是进行一个特征的非线性映射。
    """
    def __init__(self, in_features, hidden_features, dropout):
        super(Mlp, self).__init__()

        self.fc1 = nn.Linear(
            in_features,
            hidden_features,
        )

        self.fc2 = nn.Linear(
            hidden_features,
            in_features,
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def windows_partition(x, window_size):
    """
    将图像切分到window_sizexwindow_size的大小
    Args:
        x: Tensor, shape=[b, h, w, c]
        window_size: int, window size
    Returns:
        x: Tensor, shape=[num_windows*b, window_size, window_size, c]
    """
    B, H, W, C = x.shape
    x = x.reshape(
        [B, H // window_size, window_size, W // window_size, window_size,
         C])  # [bs,num_window,window_size,num_window,window_size,C]
    x = x.permute(0, 1, 3, 2, 4,
                  5)  # [bs,num_window,num_window,window_size,window_Size,C]
    x = x.reshape([-1, window_size, window_size,
                   C])  # (bs*num_windows**2,window_size, window_size, C)

    return x


def windows_reverse(windows, window_size, H, W):
    """
    将被切分的图像进行还原
    Args:
        windows: (n_windows * B, window_size, window_size, C)
        window_size: (int) window size
        H: (int) height of image
        W: (int) width of image
    Returns:
        x: (B, H, W, C)
    """

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(
        [B, H // window_size, W // window_size, window_size, window_size,
         -1])  # [bs,num_window,num_window,window_size,window_Size,C]
    x = x.permute([0, 1, 3, 2, 4,
                   5])  # [bs,num_window,window_size,num_window,window_size,C]
    x = x.reshape(
        [B, H, W,
         -1])  # (bs,num_windows*window_size, num_windows*window_size, C)
    return x


class WindowAttention(nn.Module):
    """
    基于窗口的多头自注意力机制，带有相对位置偏置，支持滑窗与不滑窗两种形式，前向传播支持传入掩码。
    Attributes:
        dim: int, input dimension (channels)
        window_size: tuple, height and width of the window
        num_heads: int, number of attention heads
        qkv_bias: bool, if True, enable learnable bias to q,k,v, default: True
        qk_scale: float, override default qk scale head_dim**-0.5 if set, default: None
        attention_dropout: float, dropout of attention
        dropout: float, dropout for output
    """
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attention_dropout=0.,
        dropout=0.,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:,
                                                                None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :,
                        0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    """
    def transpose_multihead(self, x):
        new_shape = list(x.shape[:-1]) + [self.num_heads, self.dim_head]
        x = x.reshape(new_shape)
        x = x.permute([0, 2, 1, 3])
        return x

    def get_relative_pos_bias_from_pos_index(self):
        # relative_position_bias_table is a ParamBase object
        # https://github.com/PaddlePaddle/Paddle/blob/067f558c59b34dd6d8626aad73e9943cf7f5960f/python/paddle/fluid/framework.py#L5727
        table = self.relative_position_bias_table  # N x num_heads
        # index is a tensor
        index = self.relative_position_index.reshape(
            [-1])  # window_h*window_w * window_h*window_w
        # NOTE: paddle does NOT support indexing Tensor by a Tensor
        relative_position_bias = torch.index_select(x=table, index=index)
        return relative_position_bias
    """

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin transformer block
    包含窗口多头自注意力机制，droppath，多层感知机，层标准化和残差连接
    Attributes:
        dim: int, input dimension (channels)
        input_resolution: tuple, input resoultion
        num_heads: int, number of attention heads
        window_size: int, window size, default: 7
        shift_size: int, shift size for SW-MSA, default: 0
        mlp_ratio: float, ratio of mlp hidden dim and input embedding dim, default: 4.
        qkv_bias: bool, if True, enable learnable bias to q,k,v, default: True
        qk_scale: float, override default qk scale head_dim**-0.5 if set, default: None
        dropout: float, dropout for output, default: 0.
        attention_dropout: float, dropout of attention, default: 0.
        droppath: float, drop path rate, default: 0.
    """
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = nn.LayerNorm(dim, )

        self.attn = WindowAttention(dim,
                                    window_size=(self.window_size,
                                                 self.window_size),
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attention_dropout=attention_dropout,
                                    dropout=dropout)
        self.drop_path = DropPath(droppath) if droppath > 0. else None

        self.norm2 = nn.LayerNorm(dim, )

        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       dropout=dropout)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = windows_partition(img_mask, self.window_size)
            mask_windows = mask_windows.reshape(
                (-1, self.window_size * self.window_size))
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = torch.where(
                attn_mask != 0,
                torch.ones_like(attn_mask) *
                float(-100.0),  # 这里，关于mask是否真的必要，这部分使整个代码变得复杂了极多
                attn_mask)  # 有些时候，其实我们也想结合图像边缘之间的关系
            attn_mask = torch.where(
                attn_mask == 0,  # 如果将-100设置为0网络也能work的话，Swin将大大减少代码量
                torch.zeros_like(attn_mask),
                attn_mask)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        h = x

        # x = self.norm1(x)   # [bs,H*W,C]   #后归一化，移到做完attantion之后

        new_shape = [B, H, W, C]
        x = x.reshape(new_shape)  # [bs,H,W,C]

        if self.shift_size > 0:
            shifted_x = torch.roll(x,
                                   shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))  # [bs,H,W,C]
        else:
            shifted_x = x

        x_windows = windows_partition(
            shifted_x, self.window_size)  # [bs*num_windows,7,7,C]
        x_windows = x_windows.reshape(
            [-1, self.window_size * self.window_size,
             C])  # [bs*num_windows,7*7,C]

        attn_windows = self.attn(x_windows,
                                 mask=self.attn_mask)  # [bs*num_windows,7*7,C]
        attn_windows = attn_windows.reshape(
            [-1, self.window_size, self.window_size,
             C])  # [bs*num_windows,7,7,C]

        shifted_x = windows_reverse(attn_windows, self.window_size, H,
                                    W)  # [bs,H,W,C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x,
                           shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        else:
            x = shifted_x

        x = x.reshape([B, H * W, C])  # [bs,H*W,C]
        x = self.norm1(x)  # [bs,H*W,C]

        if self.drop_path is not None:
            x = h + self.drop_path(x)
        else:
            x = h + x
        h = x  # [bs,H*W,C]

        x = self.mlp(x)  # [bs,H*W,C]
        x = self.norm2(x)

        if self.drop_path is not None:
            x = h + self.drop_path(x)
        else:
            x = h + x
        return x


class SwinT(nn.Module):
    """
    the input shape and output shape is euqal to Conv2D
    use this module can replace Conv2D by SwinT in any scene
    参数：
    in_channels: 输入通道数，同卷积
    out_channels: 输出通道数，同卷积

    以下为SwinT独有的，类似于卷积中的核大小，步幅，填充等
    input_resolution: 输入图像的尺寸大小
    num_heads: 多头注意力的头数，应该设置为能被输入通道数整除的值
    window_size: 做注意力运算的窗口的大小，窗口越大，运算就会越慢
    qkv_bias: qkv的偏置，默认None
    qk_scale: qkv的尺度，注意力大小的一个归一化，默认None      #Swin-V1版本
    dropout: 默认None
    attention_dropout: 默认None
    droppath: 默认None
    downsample: 下采样，默认False，设置为True时，输出的图片大小会变为输入的一半
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 input_resolution,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.,
                 downsample=False):
        super().__init__()
        self.dim = in_channels
        self.out_channels = out_channels
        self.input_resolution = input_resolution

        self.blocks = nn.ModuleList()
        for i in range(2):
            self.blocks.append(
                SwinTransformerBlock(dim=in_channels,
                                     input_resolution=input_resolution,
                                     num_heads=num_heads,
                                     window_size=window_size,
                                     shift_size=0 if
                                     (i % 2 == 0) else window_size // 2,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     dropout=dropout,
                                     attention_dropout=attention_dropout,
                                     droppath=droppath[i] if isinstance(
                                         droppath, list) else droppath))

        self.cnn = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )

        if downsample:
            self.downsample = PatchMerging(input_resolution,
                                           dim=in_channels,
                                           out_channels=out_channels)
        else:
            self.downsample = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape([B, C, H * W])
        x = x.permute((0, 2, 1))  # [B, H*W, C]

        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
            x = x.transpose(2, 1)  # [B, out_channels, H//2 * W//2]
            x = x.reshape([B, self.out_channels, H // 2, W // 2])
        else:
            x = x.transpose(2, 1)
            x = x.reshape([B, C, H, W])
            x = self.cnn(x)
        return x
