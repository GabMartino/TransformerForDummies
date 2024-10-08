





import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    def __init__(self, embedding_size, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.embedding_size = embedding_size
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embedding_size))
        self.beta = nn.Parameter(torch.zeros(embedding_size))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = torch.var(x, dim=-1, unbiased=False, keepdim=True) ## IT'S USED THE BIASED VARIANCE
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return x * self.gamma + self.beta


def main():
    batch_size = 1
    seq_len = 10
    embedding_size = 5
    x = torch.randn((batch_size, seq_len, embedding_size), requires_grad=False)

    layer_norm_1 = LayerNormalization(embedding_size, eps=1e-6)
    layer_norm_2 = nn.LayerNorm(embedding_size, eps=1e-6, elementwise_affine=True)
    assert torch.isclose(layer_norm_1(x), layer_norm_2(x)).all()

if __name__ == '__main__':
    main()