# 这里是SAC的编辑
# 开发时间：2025/3/10 22:09
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


# Helper functions (需要根据原 JAX 代码的 _mean_transf/_var_transf 实现补充完整)
def _mean_transf(mean, var, freqs, phases):
    """假设原实现是基于高斯分布的期望计算"""
    return torch.exp(-0.5 * freqs ** 2 * var) * torch.sin(phases + freqs * mean)


def _var_transf(mean, var, freqs, phases):
    """假设原实现是基于高斯分布的方差计算"""
    term1 = 0.5 * (1 - torch.exp(-2 * freqs ** 2 * var) * torch.cos(2 * (phases + freqs * mean)))
    term2 = _mean_transf(mean, var, freqs, phases) ** 2
    return term1 - term2


class ActLayer(nn.Module):
    """PyTorch 版本的 ActLayer"""

    def __init__(self,
                 input_dim: int,  # 新增参数，因为 PyTorch 需要预先知道输入维度
                 out_dim: int,
                 num_freqs: int,
                 use_bias: bool = True,
                 freqs_init: torch.Tensor = None,
                 phases_init: torch.Tensor = None,
                 beta_init: callable = None,
                 lamb_init: callable = None,
                 bias_init: callable = None,
                 freeze_basis: bool = False,
                 freq_scaling: bool = True,
                 freq_scaling_eps: float = 1e-3,
                 precision: str = None):
        super().__init__()

        self.out_dim = out_dim
        self.num_freqs = num_freqs
        self.input_dim = input_dim
        self.use_bias = use_bias
        self.freeze_basis = freeze_basis
        self.freq_scaling = freq_scaling
        self.freq_scaling_eps = freq_scaling_eps

        # 参数初始化逻辑
        self.freqs = Parameter(torch.empty(1, 1, 1, num_freqs))
        self.phases = Parameter(torch.empty(1, 1, 1, num_freqs))
        self.beta = Parameter(torch.empty(num_freqs, out_dim))
        self.lamb = Parameter(torch.empty(input_dim, out_dim))  # 使用预定义的 input_dim

        # 初始化参数
        freqs_init = freqs_init if freqs_init is not None else lambda x: nn.init.normal_(x, std=1.0)
        phases_init = phases_init if phases_init is not None else nn.init.zeros_
        beta_init = beta_init if beta_init is not None else lambda x: nn.init.uniform_(x, a=-math.sqrt(1 / input_dim),
                                                                                       b=math.sqrt(1 / input_dim))
        lamb_init = lamb_init if lamb_init is not None else lambda x: nn.init.uniform_(x, a=-math.sqrt(1 / input_dim),
                                                                                       b=math.sqrt(1 / input_dim))
        bias_init = bias_init if bias_init is not None else nn.init.zeros_

        freqs_init(self.freqs)
        phases_init(self.phases)
        beta_init(self.beta)
        lamb_init(self.lamb)

        if self.use_bias:
            self.bias = Parameter(torch.empty(out_dim))
            bias_init(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # 处理梯度冻结
        freqs = self.freqs.detach() if self.freeze_basis else self.freqs
        phases = self.phases.detach() if self.freeze_basis else self.phases

        # 基函数扩展
        x = torch.sin(self.freqs * x + self.phases)
        # 频率缩放
        if self.freq_scaling:
            mean = _mean_transf(0., 1., freqs, phases)
            var = _var_transf(0., 1., freqs, phases)
            x = (x - mean) / torch.sqrt(var + self.freq_scaling_eps)
        # Einsum 计算
        x = torch.einsum('bicf,fk,ck->bicf', x, self.beta, self.lamb)

        # 添加偏置
        if self.use_bias:
            x = x + self.bias

        return x


class ActLayer1(nn.Module):
    def __init__(self, input_dim, out_dim, num_freqs, use_bias=True, freqs_init=None, phases_init=None, beta_init=None, lamb_init=None, bias_init=None, freeze_basis=False, freq_scaling=True, freq_scaling_eps=1e-3):
        super().__init__()
        self.out_dim = out_dim
        self.num_freqs = num_freqs
        self.input_dim = input_dim
        self.use_bias = use_bias
        self.freeze_basis = freeze_basis
        self.freq_scaling = freq_scaling
        self.freq_scaling_eps = freq_scaling_eps

        self.freqs = Parameter(torch.empty(1, 1, 1, num_freqs))
        self.phases = Parameter(torch.empty(1, 1, 1, num_freqs))
        self.beta = Parameter(torch.empty(num_freqs, out_dim))
        self.lamb = Parameter(torch.empty(input_dim, out_dim))

        freqs_init = freqs_init if freqs_init is not None else lambda x: nn.init.uniform_(x, a=0.1, b=1.0)
        phases_init = phases_init if phases_init is not None else lambda x: nn.init.uniform_(x, a=0, b=2 * math.pi)
        beta_init = beta_init if beta_init is not None else lambda x: nn.init.xavier_uniform_(x)
        lamb_init = lamb_init if lamb_init is not None else lambda x: nn.init.xavier_uniform_(x)
        bias_init = bias_init if bias_init is not None else nn.init.zeros_

        freqs_init(self.freqs)
        phases_init(self.phases)
        beta_init(self.beta)
        lamb_init(self.lamb)

        if self.use_bias:
            self.bias = Parameter(torch.empty(out_dim))
            bias_init(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        freqs = self.freqs.detach() if self.freeze_basis else self.freqs
        phases = self.phases.detach() if self.freeze_basis else self.phases

        x = torch.sin(freqs * x + phases)

        if self.freq_scaling:
            if not hasattr(self, 'cached_mean') or not hasattr(self, 'cached_var'):
                self.cached_mean = _mean_transf(0., 1., freqs, phases)
                self.cached_var = _var_transf(0., 1., freqs, phases)
            x = (x - self.cached_mean) / torch.sqrt(self.cached_var + self.freq_scaling_eps)

        x = torch.matmul(x, self.beta)  # (batch, input_dim, num_freqs) * (num_freqs, out_dim)
        x = torch.matmul(x, self.lamb.T)  # (batch, input_dim, out_dim) * (out_dim, input_dim)

        if self.use_bias:
            x = x + self.bias

        return x


class ActNet(nn.Module):
    """PyTorch 版本的 ActNet"""

    def __init__(self,
                 input_dim: int,  # 需要明确输入维度
                 embed_dim: int,
                 num_layers: int,
                 out_dim: int,
                 num_freqs: int,
                 output_activation: callable = None,
                 op_order: str = 'A',
                 use_act_bias: bool = True,
                 freqs_init: callable = None,
                 phases_init: callable = None,
                 beta_init: callable = None,
                 lamb_init: callable = None,
                 act_bias_init: callable = None,
                 proj_bias_init: callable = None,
                 w0_init: callable = None,
                 w0_fixed: float = False,
                 freeze_basis: bool = False,
                 freq_scaling: bool = True,
                 freq_scaling_eps: float = 1e-3,
                 precision: str = None):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.num_freqs = num_freqs
        self.op_order = op_order
        self.w0_fixed = w0_fixed
        self.freq_scaling = freq_scaling
        self.freq_scaling_eps = freq_scaling_eps

        # 初始化 w0 参数
        if not self.w0_fixed:
            self.w0 = Parameter(torch.empty(1))
            nn.init.constant_(self.w0, 30.0)  # 默认初始化
        else:
            self.register_buffer('w0', torch.tensor(w0_fixed))

        # 初始投影层
        self.proj = nn.Linear(input_dim, embed_dim)
        if proj_bias_init is not None:
            proj_bias_init(self.proj.bias)

        # 构建中间层
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Sequential()
            for char in op_order:
                if char == 'A':
                    layer.append(ActLayer(
                        input_dim=embed_dim,
                        out_dim=embed_dim,
                        num_freqs=num_freqs,
                        use_bias=use_act_bias,
                        freqs_init=freqs_init,
                        phases_init=phases_init,
                        beta_init=beta_init,
                        lamb_init=lamb_init,
                        bias_init=act_bias_init,
                        freeze_basis=freeze_basis,
                        freq_scaling=freq_scaling,
                        freq_scaling_eps=freq_scaling_eps
                    ))
                elif char == 'S':
                    layer.append(nn.Identity())  # Skip 在 forward 中处理
                elif char == 'L':
                    layer.append(nn.LayerNorm(embed_dim))
            self.layers.append(layer)

        # 输出层
        self.output = nn.Linear(embed_dim, out_dim)
        nn.init.kaiming_uniform_(self.output.weight)
        self.output_activation = output_activation if output_activation is not None else lambda x: x

    def forward(self, x):
        # 初始缩放和投影
        x = x * F.softplus(self.w0) if not self.w0_fixed else x * self.w0
        # print("x_act shape:", x.shape)
        x = self.proj(x)

        # 中间层处理
        for layer in self.layers:
            y = x
            for module in layer:
                if isinstance(module, ActLayer1) or isinstance(module, nn.LayerNorm):
                    y = module(y)
                elif isinstance(module, nn.Identity):  # 处理 Skip connection
                    y = y + x
            x = y

        # 输出层
        x = self.output(x)
        return self.output_activation(x)





class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            nn.Linear(hidden_dim3, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc(x)
        return x

