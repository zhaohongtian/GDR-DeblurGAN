import torch
import torch.nn as nn
import math
# from torch.nn import init
import functools
# from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict

###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
        # norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], use_parallel=True,
             learn_residual=False):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    # if which_model_netG == 'resnet_9blocks':
    #     netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
    #                            gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    # elif which_model_netG == 'resnet_6blocks':
    #     netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6,
    #                            gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    # elif which_model_netG == 'unet_128':
    #     netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
    #                          gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    # elif which_model_netG == 'unet_256':
    #     netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
    #                          gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    # else:
    #     raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    netG = DensenetGenerator()
    # netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
    #                            gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD, n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[],
             use_parallel=True):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids, use_parallel=use_parallel)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids, use_parallel=use_parallel)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD



def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
# 加norm
def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function
# 没有norm
# def _bn_function_factory(relu, conv):
#     def bn_function(*inputs):
#         concated_features = torch.cat(inputs, 1)
#         bottleneck_output = conv(relu(concated_features))
#         return bottleneck_output
#
#     return bn_function

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.InstanceNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, growth_rate * bn_size, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.InstanceNorm2d(growth_rate * bn_size)),
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(growth_rate*bn_size, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        # 加norm
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        # 没有norm
        # bn_function = _bn_function_factory(self.relu1, self.conv1)
        if self.efficient and any(prev_features.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate = growth_rate,
                bn_size = bn_size,
                drop_rate = drop_rate,
                efficient = efficient,
            )
            self.add_module('denselayer%d' % (i+1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features,1)

class DensenetGenerator(nn.Module):
    def __init__(self, efficient=False, growth_rate=256, block_config=(9,0), compression=0.5, bn_size=4, num_init_features=256, drop_rate=0.5):
        super(DensenetGenerator,self).__init__()
        assert 0< compression <=1, 'compression of densenet should between 0 and 1'
        avgpool_size = 8

        self.features = nn.Sequential(OrderedDict([('pad0_1', nn.ReflectionPad2d(3)),
                                                   ]))
        self.features.add_module('conv0_1', nn.Conv2d(3, 64, kernel_size=7,stride=1, padding=0, bias=False))
        self.features.add_module('norm0_1', nn.InstanceNorm2d(64))
        self.features.add_module('relu0_1', nn.ReLU(inplace=True))
        # self.features.add_module('pad0_2', nn.ReflectionPad2d(1))
        self.features.add_module('conv0_2', nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False))
        self.features.add_module('norm0_2', nn.InstanceNorm2d(128))
        self.features.add_module('relu0_2', nn.ReLU(inplace=True))
        self.features.add_module('conv0_3', nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False))
        self.features.add_module('norm0_3', nn.InstanceNorm2d(256))
        self.features.add_module('relu0_3', nn.ReLU(inplace=True))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features = num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d'%(i+1), block)
            num_features = num_features + num_layers * growth_rate
            # if i != len(block_config) - 1:
            #     trans = _Transition(num_input_features=num_features, num_output_features=256)
            #     self.features.add_module('transition%d'%(i+1), trans)
            #     num_features =256
            trans = _Transition(num_input_features=num_features, num_output_features=256)
            self.features.add_module('transition%d'%(i+1), trans)
            num_features =256


        self.features.add_module('norm_final1', nn.InstanceNorm2d(num_features))
        self.features.add_module('deconv_final1', nn.ConvTranspose2d(num_features, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False))
        self.features.add_module('norm_final2', nn.InstanceNorm2d(128))
        self.features.add_module('relu_final1', nn.ReLU(inplace=True))
        self.features.add_module('deconv_final2', nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False))
        self.features.add_module('norm_final3', nn.InstanceNorm2d(64))
        self.features.add_module('relu_final2', nn.ReLU(inplace=True))
        self.features.add_module('pad_final3', nn.ReflectionPad2d(3))
        self.features.add_module('conv_final', nn.Conv2d(64, 3, kernel_size=7, padding=0))
        self.features.add_module('relu_final3', nn.Tanh())

        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        output = self.features(x)
        output = torch.clamp(x + output, min=-1, max=1)
        return output


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True,
            n_blocks=6, gpu_ids=[], use_parallel=True, a
                 =False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        self.n_blocks = n_blocks

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.BatchNorm2d
        else:
            use_bias = norm_layer == nn.BatchNorm2d

        e1_model = [
            nn.ReplicationPad2d(2),
            nn.Conv2d(input_nc, 32, kernel_size=5, padding=0, bias=use_bias),
            norm_layer(32),
            nn.ReLU(True),
            ResnetBlock(32, 32, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                use_bias=use_bias),
            ResnetBlock(32, 32, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                use_bias=use_bias)
            ]
        e2_model = [
            nn.ReflectionPad2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=2, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True),
            ResnetBlock(64, 64, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(64, 64,padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias)
        ]
        e3_model = [
            nn.ReflectionPad2d(2),
            nn.Conv2d(64, 128, kernel_size=5, padding=0, stride=2, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),
            ResnetBlock(128, 128, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(128, 128, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias)
        ]
        middle_model = [
            nn.ReflectionPad2d(2),
            nn.Conv2d(128, 256, kernel_size=5, padding=0, stride=2, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True),
            ResnetBlock(256, 256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(256, 256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True)
        ]
        d3_model = [
            ResnetBlock(128, 128, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(128, 128, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True)
        ]
        d2_model = [
            ResnetBlock(64, 64, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(64, 64, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=use_bias),
            norm_layer(32),
            nn.ReLU(True)
        ]
        d1_model = [
            ResnetBlock(32, 32, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(32, 32, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            nn.ReplicationPad2d(2),
            nn.Conv2d(32, output_nc, kernel_size=5, padding=0, bias=use_bias),
            nn.Tanh()
        ]

        self.e1_model = nn.Sequential(*e1_model)
        self.e2_model = nn.Sequential(*e2_model)
        self.e3_model = nn.Sequential(*e3_model)
        self.middle_model= nn.Sequential(*middle_model)
        self.d3_model = nn.Sequential(*d3_model)
        self.d2_model = nn.Sequential(*d2_model)
        self.d1_model = nn.Sequential(*d1_model)

    def forward(self, input):
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
        #     output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        #     output = self.model(input)
        # if self.learn_residual:
        #     # output = input + output
        #     output = torch.clamp(input + output, min=-1, max=1)
        e_1 = self.e1_model(input)
        e_2 = self.e2_model(e_1)
        e_3 = self.e3_model(e_2)
        middle = self.middle_model(e_3)
        d_3 = self.d3_model(e_3 + middle)
        d_2 = self.d2_model(d_3 + e_2)
        d_1 = self.d1_model(d_2 + e_1)
        output = d_1
        return output


# Define a resnet block
class ResnetBlock(nn.Module):

	def __init__(self, input_size, output_size,  padding_type, norm_layer, use_dropout, use_bias):
		super(ResnetBlock, self).__init__()

		padAndConv = {
			'reflect': [
                nn.ReflectionPad2d(2),
                nn.Conv2d(input_size, output_size, kernel_size=5, bias=use_bias)],
			'replicate': [
                nn.ReplicationPad2d(2),
                nn.Conv2d(input_size, output_size, kernel_size=5, bias=use_bias)],
			'zero': [
                nn.Conv2d(input_size, output_size, kernel_size=5, padding=1, bias=use_bias)]
		}

		try:
			blocks = padAndConv[padding_type] + [
				norm_layer(output_size),
				nn.ReLU(True)
            ] + [
				nn.Dropout(0.5)
			] if use_dropout else [] + padAndConv[padding_type] + [
				norm_layer(output_size)
			]
		except:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)

		self.conv_block = nn.Sequential(*blocks)

	def forward(self, x):
		out = self.conv_block(x)
		return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(
            self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d,
            use_dropout=False, gpu_ids=[], use_parallel=True, learn_residual=False):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        # currently support only input_nc == output_nc
        assert (input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer,
                                                 use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        if self.learn_residual:
            output = input + output
            output = torch.clamp(output, min=-1, max=1)
        return output


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(
            self, outer_nc, inner_nc, submodule=None,
            outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        dConv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        dRelu = nn.LeakyReLU(0.2, True)
        dNorm = norm_layer(inner_nc)
        uRelu = nn.ReLU(True)
        uNorm = norm_layer(outer_nc)

        if outermost:
            uConv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            dModel = [dConv]
            uModel = [uRelu, uConv, nn.Tanh()]
            model = [
                dModel,
                submodule,
                uModel
            ]

        elif innermost:
            uConv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            dModel = [dRelu, dConv]
            uModel = [uRelu, uConv, uNorm]
            model = [
                dModel,
                uModel
            ]

        else:
            uConv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            dModel = [dRelu, dConv, dNorm]
            uModel = [uRelu, uConv, uNorm]

            model = [
                dModel,
                submodule,
                uModel
            ]
            model += [nn.Dropout(0.5)] if use_dropout else []
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[],
                 use_parallel=True):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
