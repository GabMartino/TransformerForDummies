from html.parser import piclose
from math import trunc

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.testing.print_coercion_tables import print_new_cast_table
from torch.onnx.symbolic_opset11 import chunk


class SharedWeightsEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx = None):
        super().__init__()
        self.embeddings_size = embedding_size
        self.linear = nn.Embedding(vocab_size, embedding_size, padding_idx = padding_idx)
        self.linear_out = nn.Linear(embedding_size, vocab_size, bias=False)
        self.linear_out.weight = self.linear.weight
        assert self.linear_out.weight is self.linear.weight

    def forward(self, x):
        return self.linear(x)

    def inverse_forward(self, x):
        x = self.linear_out(x) * torch.tensor(self.embeddings_size, requires_grad=False).sqrt()
        return x


def main():
    batch_size = 64
    seq_len = 150
    vocab_size = 50000
    embedding_size = 512
    x = torch.randint(1, vocab_size ,(batch_size, seq_len), requires_grad=False)
    embedding_layer = SharedWeightsEmbedding(vocab_size=vocab_size, embedding_size=embedding_size)
    #x = x.to("cuda")
    #embedding_layer = embedding_layer.to("cuda")
    embeddings = embedding_layer(x)
    x_ = embedding_layer.inverse_forward(embeddings)
    print(x_.shape)
    x_ = torch.argmax(x_, dim=-1)
    print(x_.shape, x.shape)
    assert torch.equal(x, x_)
if __name__ == "__main__":
    main()