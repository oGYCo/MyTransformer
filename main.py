import torch
from model import Transformer

def generate_square_subsequent_mask(sz: int):
    """
    生成一个方阵掩码，其中上三角部分为 False。
    True 代表这个位置的词是可见的。
    """
    # torch.triu 会生成一个上三角矩阵
    # (torch.triu(torch.ones(sz, sz)) == 1) 会得到一个上三角为True，其余为False的矩阵
    # .transpose(0, 1) 将其变为下三角为True
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    return mask

def run_test():
    # 1. 定义超参数
    vocab_size = 10000  # 词汇表大小
    d_model = 512       # 模型维度
    num_layers = 6      # 编码器和解码器的层数
    num_heads = 8       # 多头注意力的头数
    d_ff = 2048         # 前馈网络的隐藏层维度
    dropout = 0.1

    # 2. 实例化模型
    model = Transformer(num_layers, d_model, num_heads, d_ff, vocab_size, dropout)
    print(f"模型已创建，总参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 3. 创建假数据 (batch_size=2)
    batch_size = 2
    src_len = 10 # 源序列长度
    tgt_len = 12 # 目标序列长度
    src = torch.randint(1, vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_len))

    # 4. 创建掩码
    src_mask = None # 简化处理，真实场景需要padding mask
    tgt_mask = generate_square_subsequent_mask(tgt_len)

    # 5. 模型前向传播
    print("\n开始进行一次前向传播测试...")
    output = model(src, tgt, src_mask, tgt_mask)

    # 6. 检查输出形状
    print("输入 src 形状:", src.shape)
    print("输入 tgt 形状:", tgt.shape)
    print("输出 output 形状:", output.shape)

    # 验证输出形状是否正确
    expected_shape = (batch_size, tgt_len, vocab_size)
    assert output.shape == expected_shape, f"形状不匹配! 期望得到 {expected_shape}, 实际得到 {output.shape}"
    print("\n测试通过！模型结构和数据流基本正确。")

if __name__ == '__main__':
    run_test()
