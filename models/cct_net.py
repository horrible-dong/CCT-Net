from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn

__all__ = ["cct_net_base_patch4_28", "cct_net_base_patch4_32", "cct_net_base_patch12_96",
           "cct_net_base_patch16_224", "cct_net_base_patch32_224", "cct_net_large_patch16_224",
           "cct_net_large_patch32_224", "cct_net_huge_patch14_224"]


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbedding(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Embedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_patches=196, in_c=3, embed_dim=768, num_tokens=1, drop_ratio=0.,
                 embed_layer=PatchEmbedding):
        super().__init__()
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if num_tokens == 2 else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(drop_ratio)

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        # [B, 197, 768] + [1, 197, 768] -> [B, 197, 768]
        x = self.pos_drop(x + self.pos_embed)
        return x


class Attention(nn.Module):
    def __init__(self, num_patches, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.N = num_patches + 1
        self.C = dim

    def __qkv(self, x, h: int):
        """
        :param h: the number of heads for self-attention
        """
        # x: [batch_size, num_patches + 1, total_embed_dim]
        B = x.shape[0]
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, self.N, 3, self.num_heads, self.C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]
        # [batch_size, num_heads // 2, num_patches + 1, embed_dim_per_head]
        q, k, v = (q[:, :h], q[:, h:]), (k[:, :h], k[:, h:]), (v[:, :h], v[:, h:])
        return q, k, v

    def __attn(self, q, k, v):
        # transpose: -> [batch_size, num_heads // 2, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads // 2, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        return x

    def __x(self, x1, x2):
        B = x1.shape[0]
        # cat: -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = torch.cat([x1, x2], dim=1).transpose(1, 2).reshape(B, self.N, self.C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x1, x2):
        h = 10  # 10 heads for self-attention and the other 2 heads for cross-attention

        # [batch_size, num_heads // 2, num_patches + 1, embed_dim_per_head]
        q1, k1, v1 = self.__qkv(x1, h)
        q2, k2, v2 = self.__qkv(x2, h)
        # __attn -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        x11 = self.__attn(q1[0], k1[0], v1[0])  # self-attention
        x12 = self.__attn(q1[1], k2[1], v2[1])  # cross-attention
        x22 = self.__attn(q2[0], k2[0], v2[0])  # self-attention
        x21 = self.__attn(q2[1], k1[1], v1[1])  # cross-attention

        # __x -> [batch_size, num_patches + 1, total_embed_dim]
        x1 = self.__x(x11, x12)
        x2 = self.__x(x22, x21)
        return x1, x2


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, num_patches, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1_1 = norm_layer(dim)
        self.norm1_2 = norm_layer(dim)
        self.attn = Attention(num_patches=num_patches, dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2_1 = norm_layer(dim)
        self.norm2_2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x1, x2):
        x1_, x2_ = self.attn(self.norm1_1(x1), self.norm1_2(x2))
        x1_ = x1 + self.drop_path(x1_)
        x2_ = x2 + self.drop_path(x2_)
        x1_ = x1_ + self.drop_path(self.mlp1(self.norm2_1(x1_)))
        x2_ = x2_ + self.drop_path(self.mlp2(self.norm2_2(x2_)))
        return x1_, x2_


class Sequential(nn.Sequential):
    def forward(self, x1, x2):
        for module in self:
            x1, x2 = module(x1, x2)
        return x1, x2


class CCT_Net(nn.Module):
    def __init__(self, mode, img_size=224, patch_size=16, in_c=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbedding, norm_layer=None, act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (class): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.mode = mode
        grid_size = img_size // patch_size
        num_patches = grid_size * grid_size
        num_tokens = 2 if distilled else 1
        self.num_patches = num_patches
        self.num_tokens = num_tokens
        self.embed1 = Embedding(img_size=img_size, patch_size=patch_size, num_patches=num_patches, in_c=in_c,
                                embed_dim=embed_dim, num_tokens=num_tokens, drop_ratio=drop_ratio,
                                embed_layer=embed_layer)
        self.embed2 = Embedding(img_size=img_size, patch_size=patch_size, num_patches=num_patches, in_c=in_c,
                                embed_dim=embed_dim, num_tokens=num_tokens, drop_ratio=drop_ratio,
                                embed_layer=embed_layer)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = Sequential(*[
            Block(num_patches=num_patches, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits1 = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
            self.pre_logits2 = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits1 = self.pre_logits2 = nn.Identity()

        # fc + sigmoid
        if mode == 'G':
            self.fc = nn.Sequential(
                nn.Linear(self.num_features, 4096),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(4096, 1024),
                nn.Linear(1024, 256),
                nn.Linear(256, 1)
            )
        elif mode == 'D':
            self.fc = nn.Sequential(
                nn.Linear(self.num_features * 2, 4096),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(4096, 1024),
                nn.Linear(1024, 256),
                nn.Linear(256, 1)
            )
        self.sigmoid = nn.Sigmoid()

        # Weight init
        self.apply(_init_vit_weights)

    def forward_features(self, x1, x2):
        # [B, C, H, W] -> [B, num_patches + 1, embed_dim]
        x1, x2 = self.embed1(x1), self.embed2(x2)  # [B, 197, 768]
        x1, x2 = self.blocks(x1, x2)
        x1, x2 = self.norm1(x1), self.norm2(x2)
        if self.num_tokens == 1:
            return self.pre_logits1(x1[:, 0]), self.pre_logits2(x2[:, 0])
        else:
            return (x1[:, 0], x1[:, 1]), (x2[:, 0], x2[:, 1])

    def forward(self, x, condition=None):
        # x: image_pairs [batch_size, 2, C, H, W]
        x1, x2 = x[:, 0], x[:, 1]
        x1, x2 = self.forward_features(x1, x2)
        x = torch.abs(x1 - x2)  # [B, num_features]
        if condition is not None:
            assert self.mode == 'D'
            x = torch.cat([x, condition.expand(-1, self.num_features)], dim=1)  # [B, num_features * 2]
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def cct_net_base_patch4_28(mode, has_logits: bool = False):
    model = CCT_Net(mode=mode,
                    img_size=28,
                    patch_size=4,
                    embed_dim=192,
                    depth=8,
                    num_heads=6,
                    representation_size=192 if has_logits else None)

    return model


def cct_net_base_patch4_32(mode, has_logits: bool = False):
    model = CCT_Net(mode=mode,
                    img_size=32,
                    patch_size=4,
                    embed_dim=192,
                    depth=8,
                    num_heads=8,
                    representation_size=192 if has_logits else None)

    return model


def cct_net_base_patch12_96(mode, has_logits: bool = False):
    model = CCT_Net(mode=mode,
                    img_size=96,
                    patch_size=12,
                    embed_dim=384,
                    depth=8,
                    num_heads=8,
                    representation_size=384 if has_logits else None)
    return model


def cct_net_base_patch16_224(mode, has_logits: bool = False):
    model = CCT_Net(mode=mode,
                    img_size=224,
                    patch_size=16,
                    embed_dim=768,
                    depth=12,
                    num_heads=12,
                    representation_size=768 if has_logits else None)

    return model


def cct_net_base_patch32_224(mode, has_logits: bool = False):
    model = CCT_Net(mode=mode,
                    img_size=224,
                    patch_size=32,
                    embed_dim=768,
                    depth=12,
                    num_heads=12,
                    representation_size=768 if has_logits else None)
    return model


def cct_net_large_patch16_224(mode, has_logits: bool = False):
    model = CCT_Net(mode=mode,
                    img_size=224,
                    patch_size=16,
                    embed_dim=1024,
                    depth=24,
                    num_heads=16,
                    representation_size=1024 if has_logits else None)
    return model


def cct_net_large_patch32_224(mode, has_logits: bool = False):
    model = CCT_Net(mode=mode,
                    img_size=224,
                    patch_size=32,
                    embed_dim=1024,
                    depth=24,
                    num_heads=16,
                    representation_size=1024 if has_logits else None)
    return model


def cct_net_huge_patch14_224(mode, has_logits: bool = False):
    model = CCT_Net(mode=mode,
                    img_size=224,
                    patch_size=14,
                    embed_dim=1280,
                    depth=32,
                    num_heads=16,
                    representation_size=1280 if has_logits else None)
    return model
