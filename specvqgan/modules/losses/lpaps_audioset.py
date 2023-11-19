"""
    Based on https://github.com/CompVis/taming-transformers/blob/52720829/taming/modules/losses/lpips.py
"""
from collections import namedtuple
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, '.')  # nopep8
from specvqgan.modules.losses.vggishish.model import VGGishish, VGGishish_audio
from specvqgan.util import get_ckpt_path


class LPLoss(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vggish19 features
        self.net = vggishish_audio(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="lpaps"):
        #ckpt = get_ckpt_path(name, "/mnt/autonomf_4T/text-to-sound-Synthesis-master/Codebook/specvqgan/modules/autoencoder/lpaps")
        ckpt = '/mnt/autonomf_4T/text-to-sound-Synthesis-master/Codebook/specvqgan/modules/autoencoder/lpaps/lin_vgg.pth'
        # pre_dict = torch.load(ckpt, map_location=torch.device("cpu"))
        # print('pre_dict ',pre_dict['model'].keys())
        # print('self.net ',self.net)
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False) # False ?
        # assert 1==2
        print("loaded pretrained LPAPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vggishish_lpaps"):
        if name != "vggishish_lpaps":
            raise NotImplementedError
        model = cls()
        #ckpt = get_ckpt_path(name)
        ckpt = '/mnt/autonomf_4T/text-to-sound-Synthesis-master/Codebook/specvqgan/modules/autoencoder/lpaps/lin_vgg.pth'
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        return model

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        # we are gonna use get_ckpt_path to donwload the stats as well
        stat_path = get_ckpt_path('vggishish_mean_std_melspec_10s_22050hz', '/mnt/autonomf_4T/text-to-sound-Synthesis-master/Codebook/specvqgan/modules/autoencoder/lpaps')
        # if for images we normalize on the channel dim, in spectrogram we will norm on frequency dimension
        means, stds = np.loadtxt(stat_path, dtype=np.float32).T
        # the normalization in means and stds are given for [0, 1], but specvqgan expects [-1, 1]:
        means = 2 * means - 1
        stds = 2 * stds
        # input is expected to be (B, 1, F, T)
        self.register_buffer('shift', torch.from_numpy(means)[None, None, :, None])
        self.register_buffer('scale', torch.from_numpy(stds)[None, None, :, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)

class vggishish_audio(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        vgg_pretrained_features = self.vggishish_audio(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out

    def vggishish_audio(self, pretrained: bool = True) -> VGGishish_audio:
        # loading vggishish pretrained on vggsound
        num_classes_vggsound = 527
        conv_layers = [64, 64, 'MP', 128, 128, 'MP', 256, 256, 256,256, 'MP', 512, 512, 512,512, 'MP', 512, 512, 512, 512]
        model = VGGishish_audio(conv_layers, use_bn=False, num_classes=num_classes_vggsound)
        # print('before')
        # for parameters in model.parameters():
        #     print(parameters)
        #     break
        if pretrained:
            #ckpt_path = get_ckpt_path('vggishish_lpaps', "/mnt/autonomf_4T/text-to-sound-Synthesis-master/Codebook/specvqgan/modules/autoencoder/lpaps")
            ckpt_path = '/mnt/autonomf_4T/text-to-sound-Synthesis-master/Codebook/specvqgan/modules/losses/v_logs/22-03-31T23-13-49/DataParallel-22-03-31T23-13-49.pt'
            ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
            # print(model)
            # print(ckpt['model'].keys())
            mono_model_dict = load_module2model(ckpt['model'])
            model.load_state_dict(mono_model_dict)
        # print('after')
        # for parameters in model.parameters():
        #     print(parameters)
        #     break
        # assert 1==2
        return model

def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor+eps)

def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)

def load_module2model(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items(): # k为module.xxx.weight, v为权重
        name = k[7:] # 截取`module.`后面的xxx.weight
        new_state_dict[name] = v
    return new_state_dict

if __name__ == '__main__':
    inputs = torch.rand((16, 1, 80, 848))
    reconstructions = torch.rand((16, 1, 80, 848))
    lpips = LPAPS().eval()
    loss_p = lpips(inputs.contiguous(), reconstructions.contiguous())
    # (16, 1, 1, 1)
    print(loss_p.shape)
