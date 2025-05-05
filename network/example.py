# 这里是SAC的编辑
# 开发时间：2025/3/10 22:10
# 示例参数
import torch
import torch.nn as nn
from newact import ActNet
model = ActNet(
    input_dim=64,      # 输入特征维度
    embed_dim=128,     # 嵌入维度
    num_layers=4,      # 网络层数
    out_dim=10,        # 输出维度
    num_freqs=32,      # 频率数量
    output_activation=nn.Sigmoid(),  # 输出激活
    op_order='ASL'     # 操作顺序
)

x = torch.randn(32, 64)  # 示例输入 (batch=32, dim=64)
output = model(x)