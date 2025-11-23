import torch
import torch.nn as nn
import torch.fft as fft
import argparse
from layers.RevIN import RevIN


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend



# ============================
# Sparse Top-K Selection
# ============================
class SparseTopK(nn.Module):
    def __init__(self, k_ratio=0.25):
        super().__init__()
        self.k_ratio = k_ratio

    def forward(self, x):
        B, L, C = x.shape
        k = max(1, int(L * self.k_ratio))
        topk_val, topk_idx = torch.topk(x.abs(), k, dim=1)
        mask = torch.zeros_like(x).scatter_(1, topk_idx, 1.0)
        return x * mask


# ============================
# Time-domain Sparse Mixer
# ============================
class TemporalSparseMixer(nn.Module):
    def __init__(self, dim, k_ratio):
        super().__init__()
        self.sparse = SparseTopK(k_ratio)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.proj(self.sparse(x))


# ============================
# Frequency Mixer (FFT branch)
# ============================
class FrequencyFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.enhance = nn.Linear(dim, dim)

    def forward(self, x):
        xf = fft.rfft(x, dim=1)
        xf_enh = self.enhance(xf.real) + 1j * xf.imag
        return fft.irfft(xf_enh, n=x.size(1), dim=1)


# ============================
# Sparse Cross Fusion
# ============================
class SparseCrossMixer(nn.Module):
    def __init__(self, dim, gate_init=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, num_heads=1, batch_first=True)
        self.gate = nn.Parameter(torch.ones(1) * gate_init)

    def forward(self, xt, xf):
        out, _ = self.attn(xt, xf, xf)
        return xt + torch.sigmoid(self.gate) * out


# ============================
# Upscaling Prediction Head
# ============================
class ForecastHead(nn.Module):
    def __init__(self, input_len, pred_len):
        super().__init__()
        self.linear = nn.Linear(input_len, pred_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        return x.permute(0, 2, 1)


# ============================
# SDMixer Model
# ============================
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args.enc_in
        self.time_flow = TemporalSparseMixer(dim, args.k_ratio)
        self.freq_flow = FrequencyFlow(dim)

        self.linear1 = nn.Linear(args.seq_len, args.seq_len)
        self.linear2 = nn.Linear(args.seq_len, args.seq_len)
        self.linear3 = nn.Linear(args.seq_len, args.pred_len)
        
        self.cross = SparseCrossMixer(dim, args.gate_init)
        self.head = ForecastHead(args.seq_len, args.pred_len)
        self.revin = RevIN(args.enc_in, affine=True, subtract_last=False)
        self.decomp = DFT_series_decomp()
        self.dropout = nn.Dropout(args.dropout)
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x,e0,e1,e2):
        x = self.revin(x, 'norm')
        freq,time = self.decomp(x)

        freq = freq.permute(0, 2, 1)
        time = time.permute(0, 2, 1)
        time = self.linear1(time)
        freq = self.linear2(freq)
        time = self.dropout(time)
        freq = self.dropout(freq)
        freq = freq.permute(0, 2, 1)
        time = time.permute(0, 2, 1)

        xt = self.time_flow(x)
        xf = self.freq_flow(x)
        z = self.cross(xt, xf)

        #z = self.head(self.weight * z + time + freq)
        #z = self.head(self.weight * z + freq)
        #z = self.head(self.weight * z + freq)
        z = self.revin(z, 'denorm')
        return z


# ============================
# Argparser
# ============================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=720)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--k_ratio", type=float, default=0.25)
    parser.add_argument("--gate_init", type=float, default=0.0)

    args = parser.parse_args([])
    return args


# ============================
# Demo test
# ============================
if __name__ == "__main__":
    args = get_args()
    model = Model(args)

    x = torch.randn(32, args.seq_len, args.enc_in)
    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)
