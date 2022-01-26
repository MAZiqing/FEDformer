# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils.masking import LocalMask


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    modes = min(modes, seq_len//2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index


# ##########  1d fourier layer #############
class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        print('fourier enhanced block used!')
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        print('modes={}, index={}'.format(modes, self.index))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        k = k
        v = v
        mask = mask
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)
        # batchsize = B
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x, dim=-1)
        #out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
#         if len(self.index)==0:
#             out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
#         else:
#             out_ft = torch.zeros(B, H, E, len(self.index), device=x.device, dtype=torch.cfloat)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)

        # Multiply relevant Fourier modes
        # 取guided modes的版本
        # print('x shape',x.shape)
        # print('out_ft shape',out_ft.shape)
        # print('x_ft shape',x_ft.shape)
        # print('weight shape',self.weights1.shape)
        # print('self index',self.index)
        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])

        # 取topk的modes版本
        # topk = torch.topk(torch.sum(x_ft, dim=[0, 1, 2]).abs(), dim=-1, k=self.modes1)
        # energy = (topk[0]**2).sum()
        # energy90 = 0
        # for index, j in enumerate(topk[0]):
        #     energy90 += j**2
        #     if energy90 >= energy * 0.9:
        #         break
        # for i in topk[1][:index]:
        #     out_ft[:, :, :, i] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, i])

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        #max_len = min(720,x.size(-1))
        #x = torch.fft.irfft(out_ft, n=max_len)
        # size = [B, L, H, E]
        return (x, None)


# ########## Cross Fourier Former ####################
class FourierCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random', policy=0):
        super(FourierCrossAttention, self).__init__()
        print(' fourier enhanced cross attention used!')

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)

        print('modes_q={}, index_q={}'.format(len(self.index_q), self.index_q))
        print('modes_kv={}, index_kv={}'.format(len(self.index_kv), self.index_kv))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index_q), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        mask = mask
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1) # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        #xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]

        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        xqk_ft = xqk_ft.tanh()
        #xqk_ft = torch.softmax(abs(xqk_ft),dim=-1)
        #xqk_ft = torch.complex(xqk_ft,torch.zeros_like(xqk_ft))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        #out_ft = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        #max_len = min(xq.size(-1),720)
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))
        #raise Exception('aaa')
        # size = [B, L, H, E]
        #out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=max_len)
        return (out, None)
    



