import torch
import torch.nn as nn
import math
import copy # 我们会用它来深度拷贝模块


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 的形状: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        # Q, K, V 和最终输出的线性层
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            mask = mask.unsqueeze(0) # 保证mask的维度能进行广播

        batch_size = query.size(0)

        # 1) 线性变换 Q,K,V: [batch, len, d_model] -> [batch, h, len, d_k]
        query, key, value = [
            l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = scores.softmax(dim=-1)
        p_attn = self.dropout(p_attn)

        # 3) 注意力分数与 V 相乘
        x = torch.matmul(p_attn, value)

        # 4) 拼接多头结果并做最终线性变换
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.relu(self.w_1(x))))



class EncoderLayer(nn.Module):
    def __init__(self, size: int, self_attn: MultiHeadAttention, feed_forward: PositionwiseFeedForward, dropout: float):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_connections = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(2)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.sublayer_connections[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer_connections[1](x, self.feed_forward)

class DecoderLayer(nn.Module):
    def __init__(self, size: int, self_attn: MultiHeadAttention, src_attn: MultiHeadAttention, feed_forward: PositionwiseFeedForward, dropout: float):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn # 用于与编码器输出交互的交叉注意力
        self.feed_forward = feed_forward
        self.sublayer_connections = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(3)])

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        x = self.sublayer_connections[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer_connections[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        return self.sublayer_connections[2](x, self.feed_forward)

# 辅助类：残差连接 + 层归一化 (Pre-LN 结构，更稳定)
class SublayerConnection(nn.Module):
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))



class Transformer(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        attn = MultiHeadAttention(num_heads, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        # 编码器
        encoder_layer = EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout)
        self.encoder = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

        # 解码器
        decoder_layer = DecoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(attn), copy.deepcopy(ff), dropout)
        self.decoder = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

        # 输入和输出层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = position
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoder(self.embedding(src) * math.sqrt(self.embedding.embedding_dim))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim))
        for layer in self.decoder:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.generator(x)

