# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils.masking import LocalMask
#from layers.mwt import MWT_CZ1d

def get_dynamic_modes(seq_len, modes):
    rate1 = seq_len // 96
    if rate1 <= 1:
        index = list(range(0, min(seq_len // 2, modes), 1))
    else:
        indexes = [i * seq_len / 96 for i in range(0, modes, 1)]
        indexes = [i for i in indexes if i <= seq_len // 2]
        indexes1 = list(range(0, min(seq_len//2, modes//3)))
        for i in indexes:
            if i % 1 == 0:
                indexes1 += [int(i)]
            else:
                indexes1 += [int(i)]
                indexes1 += [int(i) + 1]
        index = list(set(indexes1))
        index.sort()
    return index[:modes]


# Cross Fourier Former
class SpectralConvCross1d(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes1=0,policy=0):
        super(SpectralConvCross1d, self).__init__()
        print('corss fourier correlation used!')

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
#         self.modes1 = seq_len // 2
        self.modes1 = modes1
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        if modes1>10000:
            modes2 = modes1-10000
            self.index_q0 = list(range(0, min(seq_len_q//4, modes2//2)))
            self.index_q1 = list(range(len(self.index_q0),seq_len_q//2))
            np.random.shuffle(self.index_q1)
            self.index_q1 = self.index_q1[:min(seq_len_q//4,modes2//2)]
            self.index_q = self.index_q0+self.index_q1
            self.index_q.sort()
            
            self.index_k_v0 = list(range(0, min(seq_len_kv//4, modes2//2)))
            self.index_k_v1 = list(range(len(self.index_k_v0),seq_len_kv//2))
            np.random.shuffle(self.index_k_v1)
            self.index_k_v1 = self.index_k_v1[:min(seq_len_kv//4,modes2//2)]
            self.index_k_v = self.index_k_v0+self.index_k_v1
            self.index_k_v.sort()
            
        elif modes1 > 1000:
            modes2 = modes1-1000
            self.index_q = list(range(0, seq_len_q//2))
            np.random.shuffle(self.index_q)
            self.index_q = self.index_q[:modes2]
            self.index_q.sort()
            self.index_k_v = list(range(0, seq_len_kv // 2))
            np.random.shuffle(self.index_k_v)
            self.index_k_v = self.index_k_v[:modes2]
            self.index_k_v.sort()
        elif modes1 < 0:
            modes2 = abs(modes1)
            self.index_q = get_dynamic_modes(seq_len_q, modes2)
            self.index_k_v = list(range(0, min(seq_len_kv // 2, modes2)))
        else:
            self.index_q = list(range(0, min(seq_len_q//2, modes1)))
            self.index_k_v = list(range(0, min(seq_len_kv//2, modes1)))
            
        print('index_q={}'.format(self.index_q))
        print('len mode q={}',len(self.index_q))
        print('index_k_v={}'.format(self.index_k_v))
        print('len mode kv={}',len(self.index_k_v))

        self.register_buffer('index_q2', torch.tensor(self.index_q))
        # modes = len(index)

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index_q), dtype=torch.cfloat))

        # self.conv = nn.Conv1d(in_channels=in_channels//8, out_channels=out_channels//8, kernel_size=3)

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
        # print('xq shape',xq.shape)
        # print('xk shape',xk.shape)
        # print('xv shape',xv.shape)

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        #xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]

        xk_ft_ = torch.zeros(B, H, E, len(self.index_k_v), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_k_v):
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
    

    
class SpectralConvCross1d_local(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes1=0):
        super(SpectralConvCross1d_local, self).__init__()
        print('corss fourier correlation used!')

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
#         self.modes1 = seq_len // 2
        self.modes1 = modes1
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        if modes1>10000:
            modes2 = modes1-10000
            self.index_q0 = list(range(0, min(seq_len_q//4, modes2//2)))
            self.index_q1 = list(range(len(self.index_q0),seq_len_q//2))
            np.random.shuffle(self.index_q1)
            self.index_q1 = self.index_q1[:min(seq_len_q//4,modes2//2)]
            self.index_q = self.index_q0+self.index_q1
            self.index_q.sort()
            
            self.index_k_v0 = list(range(0, min(seq_len_kv//4, modes2//2)))
            self.index_k_v1 = list(range(len(self.index_k_v0),seq_len_kv//2))
            np.random.shuffle(self.index_k_v1)
            self.index_k_v1 = self.index_k_v1[:min(seq_len_kv//4,modes2//2)]
            self.index_k_v = self.index_k_v0+self.index_k_v1
            self.index_k_v.sort()
            
        elif modes1 > 1000:
            modes2 = modes1-1000
            self.index_q = list(range(0, seq_len_q//2))
            np.random.shuffle(self.index_q)
            self.index_q = self.index_q[:modes2]
            self.index_q.sort()
            self.index_k_v = list(range(0, seq_len_kv // 2))
            np.random.shuffle(self.index_k_v)
            self.index_k_v = self.index_k_v[:modes2]
            self.index_k_v.sort()
        elif modes1 < 0:
            modes2 = abs(modes1)
            self.index_q = get_dynamic_modes(seq_len_q, modes2)
            self.index_k_v = list(range(0, min(seq_len_kv // 2, modes2)))
        else:
            self.index_q = list(range(0, min(seq_len_q//2, modes1)))
            self.index_k_v = list(range(0, min(seq_len_kv//2, modes1)))
        print('index_q={}'.format(self.index_q))
        print('index_k_v={}'.format(self.index_k_v))

        self.register_buffer('index_q2', torch.tensor(self.index_q))
        # modes = len(index)

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index_q), dtype=torch.cfloat))
        
#         mask = self.log_mask(win_len,sub_len)
#         self.register_buffer('mask_tri',mask)

        # self.conv = nn.Conv1d(in_channels=in_channels//8, out_channels=out_channels//8, kernel_size=3)

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)
    

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        mask = mask
        B, L, H, E = q.shape
        _,S,_,_ = k.shape
        
        if L > S:
            zeros = torch.zeros_like(q[:, :(L - S), :]).float()
            v = torch.cat([v, zeros], dim=1)
            k = torch.cat([k, zeros], dim=1)
        else:
            v = v[:, :L, :, :]
            k = k[:, :L, :, :]
        
        scores = torch.einsum("blhe,bshe->bhls", q, k)
        scale = self.scale or 1. / sqrt(E)

        if mask is None:
            mask = LocalMask(B, L,L,device=q.device)

            scores.masked_fill_(mask.mask, -np.inf)

        A_local = torch.softmax(scale * scores, dim=-1)
        V_local = torch.einsum("bhls,bshd->blhd", A_local, v)
        #print(V_local)
#         print(A_local.shape)
#         print(q.shape)
#         print(V_local.shape)
        
        
        xq = q.permute(0, 2, 3, 1) # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

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
        xqk_ft = xqk_ft.tanh()
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]

        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))
        out = out.permute(0,3,1,2)
        #print(out.shape)
        out = (out+V_local)/2
        #raise Exception('aaa')
        # size = [B, L, H, E]
        return (out, None)    


################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes1=0):
        super(SpectralConv1d, self).__init__()
        print('fourier correlation used!')

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
#         self.modes1 = seq_len // 2 
        self.modes1 = modes1
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        if modes1>10000:
            modes2=modes1-10000
            self.index0 = list(range(0, min(seq_len//4, modes2//2)))
            self.index1 = list(range(len(self.index0),seq_len//2))
            np.random.shuffle(self.index1)
            self.index1 = self.index1[:min(seq_len//4,modes2//2)]
            self.index = self.index0+self.index1
            self.index.sort()
        elif modes1 > 1000:
            modes2=modes1-1000
            self.index = list(range(0, seq_len//2))
            np.random.shuffle(self.index)
            self.index = self.index[:modes2]
        else:
            self.index = list(range(0, min(seq_len//2, modes1)))

        print('modes1={}, index={}'.format(modes1, self.index))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat))
        # self.conv = nn.Conv1d(in_channels=in_channels//8, out_channels=out_channels//8, kernel_size=3)

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

    
class SpectralConv1d_local(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes1=0):
        super(SpectralConv1d_local, self).__init__()
        print('fourier correlation used!')

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
#         self.modes1 = seq_len // 2 
        self.modes1 = modes1
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        if modes1>10000:
            modes2=modes1-10000
            self.index0 = list(range(0, min(seq_len//4, modes2//2)))
            self.index1 = list(range(len(self.index0),seq_len//2))
            np.random.shuffle(self.index1)
            self.index1 = self.index1[:min(seq_len//4,modes2//2)]
            self.index = self.index0+self.index1
            self.index.sort()
        elif modes1 > 1000:
            modes2=modes1-1000
            self.index = list(range(0, seq_len//2))
            np.random.shuffle(self.index)
            self.index = self.index[:modes2]
        else:
            self.index = list(range(0, min(seq_len//2, modes1)))

        print('modes1={}, index={}'.format(modes1, self.index))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat))
        # self.conv = nn.Conv1d(in_channels=in_channels//8, out_channels=out_channels//8, kernel_size=3)

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        #break
        #print('local start')
        # size = [B, L, H, E]
        k = k
        v = v
        mask = mask
        B, L, H, E = q.shape
        
        _,S,_,_ = k.shape
        
        
        scores = torch.einsum("blhe,bshe->bhls", q, k)
        scale = self.scale or 1. / sqrt(E)

        if mask is None:
            mask = LocalMask(B, L,L,device=q.device)

            scores.masked_fill_(mask.mask, -np.inf)

        A_local = torch.softmax(scale * scores, dim=-1)
        V_local = torch.einsum("bhls,bshd->blhd", A_local, v)
        
        
        x = q.permute(0, 2, 3, 1)

        x_ft = torch.fft.rfft(x, dim=-1)

        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)

        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])

        x = torch.fft.irfft(out_ft, n=x.size(-1))
        x = (x+V_local)/2
        #print(break)

        return (x, None)

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = 16
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width)  # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
