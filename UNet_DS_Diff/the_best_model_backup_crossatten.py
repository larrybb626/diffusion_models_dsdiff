# -*- coding: utf-8 -*-
"""
@Project ：diffusion_models 
@File    ：model.py
@IDE     ：PyCharm
@Author  ：MJY
@Date    ：2024/8/26 10:09
"""
from Disc_diff.guided_diffusion.unet import SE_Attention

"""
Using disentanglement and structure guided mechanism to improve image generation    
"""
import copy

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import (TimestepEmbedSequential,
                                                      Upsample,
                                                      Downsample,
                                                      ResBlock,
                                                      AttentionBlock,
                                                      convert_module_to_f16,
                                                      convert_module_to_f32)
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.util import exists


class LeakyReLUConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            model += [
                spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        if 'norm' == 'Instance':
            model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        # elif == 'Group'

    def forward(self, x):
        return self.model(x)


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)


def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    if dim is None:
        if isinstance(module, (th.nn.ConvTranspose1d,
                               th.nn.ConvTranspose2d,
                               th.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


class SpectralNorm(object):
    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        weight_mat = weight_mat.reshape(height, -1)
        with th.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(th.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                u = F.normalize(th.matmul(weight_mat, v), dim=0, eps=self.eps)
        sigma = th.dot(u, th.matmul(weight_mat, v))
        weight = weight / sigma
        return weight, u

    def remove(self, module):
        weight = getattr(module, self.name)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, th.nn.Parameter(weight))

    def __call__(self, module, inputs):
        if module.training:
            weight, u = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, self.name + '_u', u)
        else:
            r_g = getattr(module, self.name + '_orig').requires_grad
            getattr(module, self.name).detach_().requires_grad_(r_g)

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        height = weight.size(dim)
        u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        module.register_buffer(fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_forward_pre_hook(fn)
        return fn


class MD_Dis_content(nn.Module):
    def __init__(self, c_dim=3):
        super(MD_Dis_content, self).__init__()
        model = []
        model += [LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(256, 256, kernel_size=4, stride=1, padding=0)]
        model += [nn.Conv2d(256, c_dim, kernel_size=1, stride=1, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        out = out.view(out.size(0), out.size(1))
        return out


class FeatureDisentangle(nn.Module):
    def __init__(self, in_channels, half_conv_ch):
        super(FeatureDisentangle, self).__init__()
        self.conv_1 = nn.Sequential(
            normalization(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        )
        self.conv_2 = nn.Sequential(
            normalization(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, half_conv_ch, 1, 1, 0))

    def forward(self, x):
        out = self.conv_1(x) + x
        out = self.conv_2(out)
        return out


class DSUnetModel(nn.Module):
    def __init__(self,
                 image_size,
                 in_channels,
                 model_channels,
                 out_channels,
                 num_res_blocks,
                 attention_resolutions,
                 dropout=0,
                 channel_mult=(1, 2, 4, 8),
                 conv_resample=True,
                 dims=2,
                 num_classes=None,
                 use_checkpoint=False,
                 use_fp16=False,
                 use_bf16=False,
                 num_heads=-1,
                 num_head_channels=-1,
                 num_heads_upsample=-1,
                 use_scale_shift_norm=False,
                 resblock_updown=False,
                 use_new_attention_order=False,
                 use_spatial_transformer=False,  # custom transformer support
                 transformer_depth=1,  # custom transformer support
                 context_dim=None,  # custom transformer support
                 n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
                 legacy=True,
                 disable_self_attentions=None,
                 num_attention_blocks=None,
                 disable_middle_self_attn=False,
                 use_linear_in_transformer=False,
                 adm_in_channels=None, ):
        super(DSUnetModel, self).__init__()
        if use_spatial_transformer:
            assert context_dim is not None, "context_dim must be provided for spatial transformer"
        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.dtype = th.bfloat16 if use_bf16 else self.dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        # todo try cross-attention before the middle block or in the middle block?
        # self.cross_attention_middle = TimestepEmbedSequential(
        #     SpatialTransformer(
        #         ch, num_heads, dim_head, depth=transformer_depth, context_dim=ch,
        #         disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
        #         use_checkpoint=use_checkpoint
        #     )
        # )
        # self.middle_block = TimestepEmbedSequential(
        #     ResBlock(
        #         ch,
        #         time_embed_dim,
        #         dropout,
        #         dims=dims,
        #         use_checkpoint=use_checkpoint,
        #         use_scale_shift_norm=use_scale_shift_norm,
        #     ),
        #     AttentionBlock(
        #         ch,
        #         use_checkpoint=use_checkpoint,
        #         num_heads=num_heads,
        #         num_head_channels=dim_head,
        #         use_new_attention_order=use_new_attention_order,
        #     ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
        #         ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
        #         disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
        #         use_checkpoint=use_checkpoint
        #     ),
        #     ResBlock(
        #         ch,
        #         time_embed_dim,
        #         dropout,
        #         dims=dims,
        #         use_checkpoint=use_checkpoint,
        #         use_scale_shift_norm=use_scale_shift_norm,
        #     ),
        # )
        self.cross_attention_middle = SpatialTransformer(
            ch, num_heads, dim_head, depth=4, context_dim=[ch//2]*4,
            disable_self_attn=True, use_linear=use_linear_in_transformer,
            use_checkpoint=use_checkpoint
        )
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        # self.skip_connect_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch),
                conv_nd(dims, model_channels, n_embed, 1),
                # nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
            )
        self.input_blocks_a = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        self.input_blocks_al = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        self.input_blocks_l = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        # self.middle_block_a = copy.deepcopy(self.middle_block)
        # self.middle_block_al = copy.deepcopy(self.middle_block)
        # self.middle_block_l = copy.deepcopy(self.middle_block)

        input_ch = int(channel_mult[0] * model_channels)
        # the deepest conv channel
        conv_ch = input_ch * channel_mult[-1]
        # conv silu/norm conv silu/norm silu conv
        # self.conv_style = nn.Sequential(nn.Conv2d(conv_ch, conv_ch/2, 3, 1, 1),
        #                                 nn.SiLU())
        # self.conv_content = nn.Sequential(nn.Conv2d(conv_ch, conv_ch/2, 3, 1, 1),
        #                                   nn.SiLU())
        # self.conv_str = nn.Sequential(nn.Conv2d(conv_ch/2, conv_ch/2, 3, 1, 1),
        #                            nn.SiLU())
        # self.conv_les = nn.Sequential(nn.Conv2d(conv_ch/2, conv_ch/2, 3, 1, 1),
        #                             nn.SiLU())
        half_conv_ch = int(conv_ch / 2)
        half_half_conv_ch = int(half_conv_ch / 2)
        # todo 加深网络
        self.conv_style = nn.Sequential(
            normalization(conv_ch),
            nn.SiLU(),
            nn.Conv2d(conv_ch, half_conv_ch, 1, 1, 0))
        self.conv_content = nn.Sequential(
            normalization(conv_ch),
            nn.SiLU(),
            nn.Conv2d(conv_ch, half_conv_ch, 3, 1, 1))
        self.conv_anatomy = nn.Sequential(
            nn.GroupNorm(32, conv_ch),
            nn.SiLU(),
            nn.Conv2d(conv_ch, half_conv_ch, 1, 1, 0))
        self.conv_lesion = nn.Sequential(
            nn.GroupNorm(32, conv_ch),
            nn.SiLU(),
            nn.Conv2d(conv_ch, half_conv_ch, 1, 1, 0))
        # self.conv_style = FeatureDisentangle(conv_ch, half_conv_ch)
        # self.conv_content = FeatureDisentangle(conv_ch, half_conv_ch)
        # self.conv_anatomy = FeatureDisentangle(conv_ch, half_conv_ch)
        # self.conv_lesion = FeatureDisentangle(conv_ch, half_conv_ch)
        # self.discriminator = MD_Dis_content(c_dim=3).apply(gaussian_weights_init)
        # todo Add or Concat, should change the channel number
        self.style_proj = nn.Sequential(
            SE_Attention(half_conv_ch*3, reduction=8),
            # nn.SiLU(),
            nn.Conv2d(half_conv_ch*3, half_conv_ch, 1, 1, 0)
        )
        self.share_content_proj = nn.Sequential(
            SE_Attention(half_conv_ch*3, reduction=8),
            # nn.SiLU(),
            nn.Conv2d(half_conv_ch*3, half_conv_ch, 3, 1, 1)
        )
        # self.content_proj = nn.Sequential(
        #     SE_Attention(half_conv_ch * 3, reduction=8),
        #     # nn.SiLU(),
        #     nn.Conv2d(half_conv_ch * 3, half_conv_ch, 3, 1, 1)
        # )
        # todo 这里真的要SE？或者下面真的相加？
        self.anatomy_proj = nn.Sequential(
            SE_Attention(half_conv_ch, reduction=8),
            # nn.SiLU(),
            nn.Conv2d(half_conv_ch, half_conv_ch, 3, 1, 1),
        )
        self.lesion_proj = nn.Sequential(
            SE_Attention(half_conv_ch, reduction=8),
            # nn.SiLU(),
            nn.Conv2d(half_conv_ch, half_conv_ch, 3, 1, 1)
        )
        # todo use for concat into middle block
        # self.all_proj = nn.Sequential(
        #     # normalization(conv_ch * 2),
        #     nn.SiLU(),
        #     nn.Conv2d(half_conv_ch * 8, conv_ch, 1, 1, 0),
        # )
        self.all_proj = nn.Sequential(
            # SE_Attention(half_conv_ch * 6, reduction=8),
            nn.SiLU(),
            nn.Conv2d(half_conv_ch * 6, conv_ch, 1, 1, 0),
        )
        # self.all_proj = ResBlock(
        #     conv_ch * 2,
        #     time_embed_dim,
        #     dropout,
        #     out_channels=conv_ch,
        #     dims=dims,
        #     use_checkpoint=use_checkpoint,
        #     use_scale_shift_norm=use_scale_shift_norm,
        # )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        hs_a = []
        hs_al = []
        hs_l = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        # Disentanglement
        # a: anatomy, l: lesion, al: anatomy and lesion, n: noise
        input_a = x[:, 1:2, ...]
        input_al = x[:, 2:3, ...]
        input_l = x[:, 3:4, ...]
        input_n = x[:, 0:1, ...]

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h_n = input_n.type(self.dtype)
        h_a = input_a.type(self.dtype)
        h_al = input_al.type(self.dtype)
        h_l = input_l.type(self.dtype)

        for module in self.input_blocks:
            h_n = module(h_n, emb, context)
            hs.append(h_n)
        # todo if keep the h in these input_block to plus or concat the hs , now yes
        for module in self.input_blocks_a:
            h_a = module(h_a, emb, context)
            hs_a.append(h_a)
        for module in self.input_blocks_al:
            h_al = module(h_al, emb, context)
            hs_al.append(h_al)
        for module in self.input_blocks_l:
            h_l = module(h_l, emb, context)
            hs_l.append(h_l)

        h_n = self.middle_block(h_n, emb, context)
        # h_a = self.middle_block_a(h_a, emb, context)
        # h_al = self.middle_block_al(h_al, emb, context)
        # h_l = self.middle_block_l(h_l, emb, context)
        # C-S disentangle  content: share_content
        h_n_style = self.conv_style(h_n)
        h_n_content = self.conv_content(h_n)

        h_a_style = self.conv_style(h_a)
        h_a_content = self.conv_content(h_a)

        h_al_style = self.conv_style(h_al)
        h_al_content = self.conv_content(h_al)

        h_l_style = self.conv_style(h_l)
        h_l_content = self.conv_content(h_l)
        # lesion anatomy disentangle

        h_a_anatomy = self.conv_anatomy(h_a)
        h_al_anatomy = self.conv_anatomy(h_al)

        h_al_lesion = self.conv_lesion(h_al)
        h_l_lesion = self.conv_lesion(h_l)

        h_style_list = [h_a_style, h_al_style, h_l_style]  # 独有的风格
        h_content_list = [h_a_content, h_al_content, h_l_content]  # 共享的内容
        h_anatomy_list = [h_a_anatomy, h_al_anatomy]  # 独有的结构
        h_lesion_list = [h_al_lesion, h_l_lesion]  # 独有的病变

        # todo Add or Concat
        h_style = self.style_proj(th.cat(h_style_list[:], dim=1))
        h_share_content = self.share_content_proj(th.cat(h_content_list[:], dim=1))
        # h_style = self.style_proj(th.mean(th.stack(h_style_list[:]), dim=0))
        # h_share_content = self.share_content_proj(th.mean(th.stack(h_content_list[:]), dim=0))
        h_anatomy = self.anatomy_proj(th.mean(th.stack(h_anatomy_list), dim=0))
        h_lesion = self.lesion_proj(th.mean(th.stack(h_lesion_list), dim=0))
        # h_anatomy = self.anatomy_proj(th.cat(h_anatomy_list, dim=1))
        # h_lesion = self.lesion_proj(th.cat(h_lesion_list, dim=1))
        # h_content = self.content_proj(th.cat([h_share_content, h_anatomy, h_lesion], dim=1))
        h_n_and_all_list = [h_style, h_n_style, h_share_content, h_n_content]

        # feature_scal =

        # todo first experiment concat into middle block, later maybe attention
        # h = th.cat([h_n, h_share_content, h_style, h_anatomy, h_lesion], dim=1)
        # h = th.cat([h_n, h_a, h_al, h_l], dim=1)
        #
        # h = self.all_proj(h, emb)
        # h = self.all_proj(h)
        context = [h_share_content, h_style, h_anatomy, h_lesion]
        h = self.cross_attention_middle(h_n, context)
        #

        for module in self.output_blocks:
            # todo cat or plus
            h = th.cat([h, (hs.pop() + hs_a.pop() + hs_al.pop() + hs_l.pop()) / 4], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return (self.out(h),
                    {"style": h_style_list,
                     "content": h_content_list,
                     "anatomy": h_anatomy_list,
                     "lesion": h_lesion_list,
                     "n_style_content": h_n_and_all_list, })
