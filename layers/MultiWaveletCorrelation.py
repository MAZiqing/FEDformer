import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import List, Tuple
import math
from functools import partial
from einops import rearrange, reduce, repeat
from torch import nn, einsum, diagonal
from math import log2, ceil
import pdb
from utils.masking import LocalMask
from layers.utils import get_filter


from layers.FourierCorrelation import FourierBlock, FourierCrossAttention


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ##
class mwt_transform(nn.Module):
    def __init__(self, ich=1, k=8, alpha=16, c=128, nCZ=1,
                 L=0,
                 base='legendre', attention_dropout=0.1):
        super(mwt_transform, self).__init__()
        print('base', base)
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ
        self.Lk0 = nn.Linear(ich, c * k)
        self.Lk1 = nn.Linear(c * k, ich)
        self.ich = ich
        self.MWT_CZ = nn.ModuleList(MWT_CZ1d(k, alpha, L, c, base) for i in range(nCZ))

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]
        values = values.view(B, L, -1)

        V = self.Lk0(values).view(B, L, self.c, -1)
        for i in range(self.nCZ):
            V = self.MWT_CZ[i](V)
            if i < self.nCZ - 1:
                V = F.relu(V)

        V = self.Lk1(V.view(B, L, -1))
        V = V.view(B, L, -1, D)
        return (V.contiguous(), None)


class FourierCrossAttention1(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=16, activation='tanh',
                 mode_select_method='random'):
        super(FourierCrossAttention1, self).__init__()
        print('corss fourier correlation used!')

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        #         self.modes1 = seq_len // 2
        self.modes1 = modes
        self.activation = activation

        if modes > 10000:
            modes2 = modes - 10000
            self.index_q0 = list(range(0, min(seq_len_q // 4, modes2 // 2)))
            self.index_q1 = list(range(len(self.index_q0), seq_len_q // 2))
            np.random.shuffle(self.index_q1)
            self.index_q1 = self.index_q1[:min(seq_len_q // 4, modes2 // 2)]
            self.index_q = self.index_q0 + self.index_q1
            self.index_q.sort()

            self.index_k_v0 = list(range(0, min(seq_len_kv // 4, modes2 // 2)))
            self.index_k_v1 = list(range(len(self.index_k_v0), seq_len_kv // 2))
            np.random.shuffle(self.index_k_v1)
            self.index_k_v1 = self.index_k_v1[:min(seq_len_kv // 4, modes2 // 2)]
            self.index_k_v = self.index_k_v0 + self.index_k_v1
            self.index_k_v.sort()

        # elif modes > 1000:
        #     modes2 = modes - 1000
        #     self.index_q = list(range(0, seq_len_q // 2))
        #     np.random.shuffle(self.index_q)
        #     self.index_q = self.index_q[:modes2]
        #     self.index_q.sort()
        #     self.index_k_v = list(range(0, seq_len_kv // 2))
        #     np.random.shuffle(self.index_k_v)
        #     self.index_k_v = self.index_k_v[:modes2]
        #     self.index_k_v.sort()
        # elif modes < 0:
        #     modes2 = abs(modes)
        #     self.index_q = get_dynamic_modes(seq_len_q, modes2)
        #     self.index_k_v = list(range(0, min(seq_len_kv // 2, modes2)))
        # else:
        #     self.index_q = list(range(0, min(seq_len_q // 2, modes)))
        #     self.index_k_v = list(range(0, min(seq_len_kv // 2, modes)))

        print('index_q={}'.format(self.index_q))
        print('len mode q={}', len(self.index_q))
        print('index_k_v={}'.format(self.index_k_v))
        print('len mode kv={}', len(self.index_k_v))

        self.register_buffer('index_q2', torch.tensor(self.index_q))

    #         self.scale = (1 / (in_channels * out_channels))
    #         self.weights1 = nn.Parameter(
    #             self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index_q), dtype=torch.cfloat))

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        mask = mask
        B, L, E, H = q.shape

        xq = q.permute(0, 3, 2, 1)  # size = [B, H, E, L] torch.Size([3, 8, 64, 512])
        xk = k.permute(0, 3, 2, 1)
        xv = v.permute(0, 3, 2, 1)
        self.index_q = list(range(0, min(int(L // 2), self.modes1)))
        self.index_k_v = list(range(0, min(int(xv.shape[3] // 2), self.modes1)))

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]

        xk_ft_ = torch.zeros(B, H, E, len(self.index_k_v), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_k_v):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]
        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        # print('xqkv_ft',xqkv_ft.shape)
        # print('self.weights1',self.weights1.shape)
        #         xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        xqkvw = xqkv_ft
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]

        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1)).permute(0, 3, 2, 1)
        # raise Exception('aaa')
        # size = [B, L, H, E]
        return (out, None)


def get_initializer(name):
    if name == 'xavier_normal':
        init_ = partial(nn.init.xavier_normal_)
    elif name == 'kaiming_uniform':
        init_ = partial(nn.init.kaiming_uniform_)
    elif name == 'kaiming_normal':
        init_ = partial(nn.init.kaiming_normal_)
    return init_


def compl_mul1d(x, weights):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", x, weights)


# class Sparsemax(nn.Module):
#     """Sparsemax function."""
#
#     def __init__(self, dim=None):
#         """Initialize sparsemax activation
#
#         Args:
#             dim (int, optional): The dimension over which to apply the sparsemax function.
#         """
#         super(Sparsemax, self).__init__()
#
#         self.dim = -1 if dim is None else dim
#
#     def forward(self, input):
#         """Forward function.
#         Args:
#             input (torch.Tensor): Input tensor. First dimension should be the batch size
#         Returns:
#             torch.Tensor: [batch_size x number_of_logits] Output tensor
#         """
#         # Sparsemax currently only handles 2-dim tensors,
#         # so we reshape to a convenient shape and reshape back after sparsemax
#         input = input.transpose(0, self.dim)
#         original_size = input.size()
#         input = input.reshape(input.size(0), -1)
#         input = input.transpose(0, 1)
#         dim = 1
#
#         number_of_logits = input.size(dim)
#
#         # Translate input by max for numerical stability
#         input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)
#
#         # Sort input in descending order.
#         # (NOTE: Can be replaced with linear time selection method described here:
#         # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
#         zs = torch.sort(input=input, dim=dim, descending=True)[0]
#         range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=device, dtype=input.dtype).view(1, -1)
#         range = range.expand_as(zs)
#
#         # Determine sparsity of projection
#         bound = 1 + range * zs
#         cumulative_sum_zs = torch.cumsum(zs, dim)
#         is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
#         k = torch.max(is_gt * range, dim, keepdim=True)[0]
#
#         # Compute threshold function
#         zs_sparse = is_gt * zs
#
#         # Compute taus
#         taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
#         taus = taus.expand_as(input)
#
#         # Sparsemax
#         self.output = torch.max(torch.zeros_like(input), input - taus)
#
#         # Reshape back to original shape
#         output = self.output
#         output = output.transpose(0, 1)
#         output = output.reshape(original_size)
#         output = output.transpose(0, self.dim)
#
#         return output
#
#     def backward(self, grad_output):
#         """Backward function."""
#         dim = 1
#
#         nonzeros = torch.ne(self.output, 0)
#         sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
#         self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))
#
#         return self.grad_input


class sparseKernelFT1d(nn.Module):
    def __init__(self,
                 k, alpha, c=1,
                 nl=1,
                 initializer=None,
                 **kwargs):
        super(sparseKernelFT1d, self).__init__()

        self.modes1 = alpha
        self.scale = (1 / (c * k * c * k))
        self.weights1 = nn.Parameter(self.scale * torch.rand(c * k, c * k, self.modes1, dtype=torch.cfloat))
        self.weights1.requires_grad = True
        self.k = k

    def forward(self, x):
        B, N, c, k = x.shape  # (B, N, c, k)

        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)
        x_fft = torch.fft.rfft(x)
        # Multiply relevant Fourier modes
        l = min(self.modes1, N // 2 + 1)
        # l = N//2+1
        out_ft = torch.zeros(B, c * k, N // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :l] = compl_mul1d(x_fft[:, :, :l], self.weights1[:, :, :l])
        x = torch.fft.irfft(out_ft, n=N)
        x = x.permute(0, 2, 1).view(B, N, c, k)
        return x

# ##
class MWT_CZ1d(nn.Module):
    def __init__(self,
                 k=3, alpha=64,
                 L=0, c=1,
                 base='legendre',
                 initializer=get_initializer('xavier_normal'),
                 **kwargs):
        super(MWT_CZ1d, self).__init__()

        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3

        self.A = sparseKernelFT1d(k, alpha, c)
        self.B = sparseKernelFT1d(k, alpha, c)
        self.C = sparseKernelFT1d(k, alpha, c)

        self.T0 = nn.Linear(k, k)

        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((G0.T, G1.T), axis=0)))

        self.register_buffer('rc_e', torch.Tensor(
            np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(
            np.concatenate((H1r, G1r), axis=0)))

    def forward(self, x):

        B, N, c, k = x.shape  # (B, N, k)

        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_x = x[:, 0:nl - N, :, :]
        x = torch.cat([x, extra_x], 1)
        # print('x shape raw',x.shape)
        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])
        #         decompose
        for i in range(ns - self.L):
            # print('x shape',x.shape)
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]

        # print('x shape decomposed',x.shape)
        x = self.T0(x)  # coarsest scale transform

        #        reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            # print('Us {} shape {}'.format(i,Us[i].shape))
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            # print('reconsturct step {} shape {}'.format(i,x.shape))
            x = self.evenOdd(x)
        # raise Exception('test break')
        x = x[:, :N, :, :]
        # print('new x shape',x.shape)
        # raise Exception('break')

        return x

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :],
                        x[:, 1::2, :, :],
                        ], -1)
        # print('x shape',x.shape)
        # print('xa shape',xa.shape)
        # print('ec_d shape',self.ec_d.shape)
        # print('ec_s shape',self.ec_s.shape)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):

        B, N, c, ich = x.shape  # (B, N, c, k)
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x = torch.zeros(B, N * 2, c, self.k,
                        device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x

# ##
class MWT_CZ1d_cross(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes, c=64,
                 k=8, ich=512,
                 L=0,
                 base='legendre',
                 mode_select_method='random',
                 initializer=get_initializer('xavier_normal'), activation='tanh',
                 **kwargs):
        super(MWT_CZ1d_cross, self).__init__()
        print('base', base)

        self.c = c

        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3

        self.attn1 = FourierCrossAttention(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                           seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                           mode_select_method=mode_select_method)
        self.attn2 = FourierCrossAttention(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                           seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                           mode_select_method=mode_select_method)
        self.attn3 = FourierCrossAttention(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                           seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                           mode_select_method=mode_select_method)
        self.attn4 = FourierCrossAttention(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                           seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                           mode_select_method=mode_select_method)

        self.T0 = nn.Linear(k, k)

        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((G0.T, G1.T), axis=0)))

        self.register_buffer('rc_e', torch.Tensor(
            np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(
            np.concatenate((H1r, G1r), axis=0)))

        self.Lk = nn.Linear(ich, c * k)
        self.Lq = nn.Linear(ich, c * k)
        self.Lv = nn.Linear(ich, c * k)
        self.out = nn.Linear(c * k, ich)
        self.modes1 = modes

    def forward(self, q, k, v, mask=None):
        B, N, H, E = q.shape  # (B, N, k)
        _, S, _, _ = k.shape

        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)
        v = v.view(v.shape[0], v.shape[1], -1)
        q = self.Lq(q)
        q = q.view(q.shape[0], q.shape[1], self.c, self.k)
        k = self.Lk(k)
        k = k.view(k.shape[0], k.shape[1], self.c, self.k)
        v = self.Lv(v)
        v = v.view(v.shape[0], v.shape[1], self.c, self.k)

        if N > S:
            zeros = torch.zeros_like(q[:, :(N - S), :]).float()
            v = torch.cat([v, zeros], dim=1)
            k = torch.cat([k, zeros], dim=1)
        else:
            v = v[:, :N, :, :]
            k = k[:, :N, :, :]

        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_q = q[:, 0:nl - N, :, :]
        extra_k = k[:, 0:nl - N, :, :]
        extra_v = v[:, 0:nl - N, :, :]
        q = torch.cat([q, extra_q], 1)
        k = torch.cat([k, extra_k], 1)
        v = torch.cat([v, extra_v], 1)

        Ud_q = torch.jit.annotate(List[Tuple[Tensor]], [])
        Ud_k = torch.jit.annotate(List[Tuple[Tensor]], [])
        Ud_v = torch.jit.annotate(List[Tuple[Tensor]], [])

        Us_q = torch.jit.annotate(List[Tensor], [])
        Us_k = torch.jit.annotate(List[Tensor], [])
        Us_v = torch.jit.annotate(List[Tensor], [])

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

        #         decompose
        for i in range(ns - self.L):
            # print('q shape',q.shape)
            d, q = self.wavelet_transform(q)
            Ud_q += [tuple([d, q])]
            Us_q += [d]
        for i in range(ns - self.L):
            d, k = self.wavelet_transform(k)
            Ud_k += [tuple([d, k])]
            Us_k += [d]
        for i in range(ns - self.L):
            d, v = self.wavelet_transform(v)
            Ud_v += [tuple([d, v])]
            Us_v += [d]
        for i in range(ns - self.L):
            dk, sk = Ud_k[i], Us_k[i]
            dq, sq = Ud_q[i], Us_q[i]
            dv, sv = Ud_v[i], Us_v[i]
            # Ud += [self.attn1(dq[0].transpose(2, 3), dk[0].transpose(2, 3), dv[0].transpose(2, 3), mask)[0]
            #        + self.attn2(dq[1].transpose(1, 2), dk[1].transpose(1, 2), dv[1].transpose(1, 2), mask)[0]]
            Ud += [self.attn1(dq[0], dk[0], dv[0], mask)[0] + self.attn2(dq[1], dk[1], dv[1], mask)[0]]
            # Us += [self.attn3(sq.transpose(1, 2), sk.transpose(1, 2), sv.transpose(1, 2), mask)[0]]
            Us += [self.attn3(sq, sk, sv, mask)[0]]
        # v = self.attn4(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mask)[0]
        v = self.attn4(q, k, v, mask)[0]

        #        reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            v = v + Us[i]
            v = torch.cat((v, Ud[i]), -1)
            v = self.evenOdd(v)
        # raise Exception('test break')
        v = self.out(v[:, :N, :, :].contiguous().view(B, N, -1))
        return (v.contiguous(), None)

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :],
                        x[:, 1::2, :, :],
                        ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):

        B, N, c, ich = x.shape  # (B, N, c, k)
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x = torch.zeros(B, N * 2, c, self.k,
                        device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x
