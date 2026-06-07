from IPython.core.debugger import set_trace
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math


# ===========================================================================
# Original LayerNorm (kept for backward compatibility)
# ===========================================================================
class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):
        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)
            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs**2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 2
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


# ===========================================================================
# Tropical LayerNorm
# ===========================================================================
class TropicalLayerNorm(nn.Module):
    """
    Tropical LayerNorm — Tích hợp ánh xạ Piecewise-Linear (PWL) vào LayerNorm.

    Ba bước toán học:
        1. Euclidean Normalization:
               x_std = (x - mean) / (std + eps)
        2. Piecewise-Linear / Tropical mapping:
               f_trop(x_std) = a_k * x_std + b_k   if x_std ∈ [t_k, t_{k+1})
           Trong đó {t_k} là các breakpoints học được,
           a_k (slopes) và b_k (piecewise_biases) là tham số học được.
        3. Affine Transform (pretrained-compatible):
               y = gamma ⊙ f_trop(x_std) + beta

    Bảo toàn tri thức tại khởi tạo (t=0):
        a_k = 1, b_k = 0  →  f_trop = identity  →  TropicalLN ≡ LayerNorm gốc

    Hỗ trợ Conditional LayerNorm (conditional=True):
        gamma và beta được điều chỉnh bởi vector điều kiện cond.

    Args:
        input_dim   : chiều đặc trưng đầu vào (d)
        num_pieces  : số phân vùng tuyến tính K (mặc định 4)
        cond_dim    : chiều vector điều kiện (chỉ dùng khi conditional=True)
        conditional : nếu True, gamma/beta phụ thuộc vào cond
        epsilon     : hằng số tránh chia 0
        value_range : phạm vi giá trị của x_std đặt breakpoints (mặc định [-3, 3])
    """

    def __init__(self,
                 input_dim: int,
                 num_pieces: int = 4,
                 cond_dim: int = 0,
                 conditional: bool = False,
                 epsilon: float = 1e-12,
                 value_range: tuple = (-3.0, 3.0)):
        super(TropicalLayerNorm, self).__init__()

        self.input_dim = input_dim
        self.num_pieces = num_pieces          # K
        self.cond_dim = cond_dim
        self.conditional = conditional
        self.epsilon = epsilon
        self.value_range = value_range        # domain of x_std for breakpoints

        # ------------------------------------------------------------------ #
        # Bước 3 — Affine parameters (gamma, beta), khởi tạo như LayerNorm   #
        # ------------------------------------------------------------------ #
        self.gamma = Parameter(torch.ones(input_dim))   # scale
        self.beta  = Parameter(torch.zeros(input_dim))  # shift

        # ------------------------------------------------------------------ #
        # Bước 2 — Tham số Tropical / PWL                                    #
        #   breakpoints : (K-1,) — ranh giới phân vùng (học được, shared)    #
        #   slopes      : (K,)   — độ dốc mỗi phân vùng                      #
        #   pw_biases   : (K,)   — độ lệch mỗi phân vùng                     #
        # ------------------------------------------------------------------ #
        # Khởi tạo K-1 breakpoints đều nhau trong value_range
        lo, hi = value_range
        init_bps = torch.linspace(lo, hi, num_pieces + 1)[1:-1]  # (K-1,)
        self.breakpoints = Parameter(init_bps)

        # Khởi tạo slopes=1, biases=0 → identity → bảo toàn tri thức
        self.slopes     = Parameter(torch.ones(num_pieces))   # a_k
        self.pw_biases  = Parameter(torch.zeros(num_pieces))  # b_k

        # ------------------------------------------------------------------ #
        # Conditional: gamma/beta được cộng thêm ánh xạ từ cond              #
        # ------------------------------------------------------------------ #
        if self.conditional:
            assert cond_dim > 0, "cond_dim phải > 0 khi conditional=True"
            self.gamma_dense = nn.Linear(cond_dim, input_dim, bias=False)
            self.beta_dense  = nn.Linear(cond_dim, input_dim, bias=False)
            # Khởi tạo 0 để không làm nhiễu pretrained weights ban đầu
            nn.init.constant_(self.gamma_dense.weight, 0.0)
            nn.init.constant_(self.beta_dense.weight,  0.0)

    # ---------------------------------------------------------------------- #
    # Piecewise-Linear (Tropical) mapping — vectorised, no Python loop        #
    # ---------------------------------------------------------------------- #
    def _tropical_map(self, x_std: torch.Tensor) -> torch.Tensor:
        """
        x_std: (..., d) — đã chuẩn hoá Euclidean
        Trả về f_trop(x_std): (..., d)

        Sử dụng softmax-gated mixture thay vì hard if-else để giữ
        gradient flow qua tất cả các phân vùng (differentiable):

            gate_k = softmax(−|x_std − c_k| / τ)_k
            f_trop = Σ_k gate_k * (a_k * x_std + b_k)

        Trong đó c_k là trung điểm của phân vùng k, τ là nhiệt độ.
        Khi τ→0, hội tụ về hard piecewise-linear.
        """
        # Tính trung điểm các phân vùng từ breakpoints đã học
        lo = torch.full((1,), self.value_range[0], device=x_std.device, dtype=x_std.dtype)
        hi = torch.full((1,), self.value_range[1], device=x_std.device, dtype=x_std.dtype)

        # breakpoints đã sắp xếp để đảm bảo tính nhất quán
        sorted_bps, _ = torch.sort(self.breakpoints)          # (K-1,)
        boundaries = torch.cat([lo, sorted_bps, hi], dim=0)   # (K+1,)
        centers = (boundaries[:-1] + boundaries[1:]) / 2.0    # (K,)

        # x_std: (..., d), centers: (K,) — broadcasting
        # dist: (..., d, K)
        x_exp = x_std.unsqueeze(-1)           # (..., d, 1)
        dist  = torch.abs(x_exp - centers)    # (..., d, K)

        # Soft gate — nhiệt độ τ = 0.5 (có thể tune)
        tau   = 0.5
        gates = torch.softmax(-dist / tau, dim=-1)  # (..., d, K)

        # slopes: (K,), pw_biases: (K,)
        # piece_out_k = a_k * x_std + b_k → (..., d, K)
        piece_out = x_exp * self.slopes + self.pw_biases   # broadcast

        # Mixture → (..., d)
        f_trop = (gates * piece_out).sum(dim=-1)
        return f_trop

    def forward(self, inputs: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        """
        inputs : (..., input_dim)
        cond   : (..., cond_dim) — chỉ dùng khi conditional=True
        """
        # ------------------------------------------------------------------ #
        # Bước 1: Euclidean Normalization                                     #
        # ------------------------------------------------------------------ #
        mean    = inputs.mean(dim=-1, keepdim=True)             # (..., 1)
        var     = inputs.var(dim=-1, keepdim=True, unbiased=False)
        x_std   = (inputs - mean) / (var + self.epsilon).sqrt() # (..., d)

        # ------------------------------------------------------------------ #
        # Bước 2: Tropical / Piecewise-Linear mapping                         #
        # ------------------------------------------------------------------ #
        f_trop = self._tropical_map(x_std)   # (..., d)

        # ------------------------------------------------------------------ #
        # Bước 3: Affine Transform                                            #
        # ------------------------------------------------------------------ #
        if self.conditional and cond is not None:
            # Broadcast cond lên cùng số chiều với inputs
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)
            gamma = self.gamma + self.gamma_dense(cond)  # (..., d)
            beta  = self.beta  + self.beta_dense(cond)   # (..., d)
        else:
            gamma = self.gamma   # (d,)
            beta  = self.beta    # (d,)

        outputs = gamma * f_trop + beta
        return outputs


# ===========================================================================
# HandshakingKernel — mở rộng thêm tropical_cln, tropical_cln_plus
# ===========================================================================
class HandshakingKernel(nn.Module):
    """
    Shaking_type options:
        cat              — concatenate + linear
        cat_plus         — cat + inner context + linear
        cln              — Conditional LayerNorm (original)
        cln_plus         — Conditional LayerNorm + inner context (original)
        tropical_cln     — Tropical Conditional LayerNorm (NEW)
        tropical_cln_plus— Tropical CLN + inner context (NEW)
    """

    def __init__(self, hidden_size, shaking_type, inner_enc_type, tropical_num_pieces=4):
        """
        hidden_size        : kích thước hidden
        shaking_type       : loại kết hợp token-pair
        inner_enc_type     : loại encoder bên trong (lstm, mean_pooling, ...)
        tropical_num_pieces: số phân vùng K cho TropicalLayerNorm
        """
        super().__init__()
        self.shaking_type = shaking_type

        if shaking_type == "cat":
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == "cat_plus":
            self.combine_fc = nn.Linear(hidden_size * 3, hidden_size)

        # Original CLN
        elif shaking_type == "cln":
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional=True)
        elif shaking_type == "cln_plus":
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional=True)
            self.inner_context_cln = LayerNorm(hidden_size, hidden_size, conditional=True)

        # ------------------------------------------------------------------ #
        # Tropical CLN (NEW)                                                  #
        # ------------------------------------------------------------------ #
        elif shaking_type == "tropical_cln":
            self.tp_cln = TropicalLayerNorm(
                input_dim=hidden_size,
                num_pieces=tropical_num_pieces,
                cond_dim=hidden_size,
                conditional=True,
            )
        elif shaking_type == "tropical_cln_plus":
            self.tp_cln = TropicalLayerNorm(
                input_dim=hidden_size,
                num_pieces=tropical_num_pieces,
                cond_dim=hidden_size,
                conditional=True,
            )
            self.inner_context_cln = TropicalLayerNorm(
                input_dim=hidden_size,
                num_pieces=tropical_num_pieces,
                cond_dim=hidden_size,
                conditional=True,
            )
        else:
            raise ValueError(
                f"shaking_type '{shaking_type}' không hợp lệ. "
                "Chọn một trong: cat, cat_plus, cln, cln_plus, tropical_cln, tropical_cln_plus"
            )

        self.inner_enc_type = inner_enc_type
        if inner_enc_type == "mix_pooling":
            self.lamtha = Parameter(torch.rand(hidden_size))
        elif inner_enc_type == "lstm":
            self.inner_context_lstm = nn.LSTM(
                hidden_size, hidden_size,
                num_layers=1, bidirectional=False, batch_first=True
            )

    def enc_inner_hiddens(self, seq_hiddens, inner_enc_type="lstm"):
        """seq_hiddens: (batch_size, seq_len, hidden_size)"""
        def pool(seqence, pooling_type):
            if pooling_type == "mean_pooling":
                return torch.mean(seqence, dim=-2)
            elif pooling_type == "max_pooling":
                pooling, _ = torch.max(seqence, dim=-2)
                return pooling
            elif pooling_type == "mix_pooling":
                return (self.lamtha * torch.mean(seqence, dim=-2)
                        + (1 - self.lamtha) * torch.max(seqence, dim=-2)[0])

        if "pooling" in inner_enc_type:
            inner_context = torch.stack(
                [pool(seq_hiddens[:, :i+1, :], inner_enc_type) for i in range(seq_hiddens.size()[1])],
                dim=1
            )
        elif inner_enc_type == "lstm":
            inner_context, _ = self.inner_context_lstm(seq_hiddens)

        return inner_context

    def forward(self, seq_hiddens):
        """
        seq_hiddens: (batch_size, seq_len, hidden_size)
        returns:
            shaking_hiddens: (batch_size, seq_len*(seq_len+1)//2, hidden_size)
        """
        seq_len = seq_hiddens.size()[-2]
        shaking_hiddens_list = []

        for ind in range(seq_len):
            hidden_each_step = seq_hiddens[:, ind, :]                         # (B, H)
            visible_hiddens  = seq_hiddens[:, ind:, :]                         # (B, L-ind, H)
            repeat_hiddens   = hidden_each_step[:, None, :].repeat(1, seq_len - ind, 1)  # (B, L-ind, H)

            if self.shaking_type == "cat":
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))

            elif self.shaking_type == "cat_plus":
                inner_context   = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens, inner_context], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))

            elif self.shaking_type == "cln":
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)

            elif self.shaking_type == "cln_plus":
                inner_context   = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
                shaking_hiddens = self.inner_context_cln(shaking_hiddens, inner_context)

            # ---------------------------------------------------------------- #
            # Tropical CLN (NEW)                                               #
            # ---------------------------------------------------------------- #
            elif self.shaking_type == "tropical_cln":
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)

            elif self.shaking_type == "tropical_cln_plus":
                inner_context   = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
                shaking_hiddens = self.inner_context_cln(shaking_hiddens, inner_context)

            shaking_hiddens_list.append(shaking_hiddens)

        long_shaking_hiddens = torch.cat(shaking_hiddens_list, dim=1)
        return long_shaking_hiddens