import math

import torch

import torch.nn as nn
from PIL.ImageOps import scale
from matplotlib import pyplot as plt

from model.utils.utils import create_random_padding_mask, create_look_ahead_mask
import logging

def create_look_ahead_mask(seq_len, device):
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)


def scaled_dot_attention(q, k, v, mask = None, dropout = None, isCausal = False):
    logging.debug("Q= ", q )
    scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if mask is not None:
        mask = mask.unsqueeze(1) ## extend for the num_heads
        scaled.masked_fill_(mask.bool(), float('-inf'))
        assert not torch.isnan(scaled).any()
    if isCausal:
        look_ahead_mask = create_look_ahead_mask(scaled.size(-1), device=scaled.device)
        scaled.masked_fill_(look_ahead_mask.bool(), float('-inf'))

    attention = torch.softmax(scaled, dim=-1)
    if dropout is not None:
        attention = torch.dropout(attention, p=dropout)
    output = torch.matmul(attention, v)
    return output, attention

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout_rate = None):
        super(MultiHeadSelfAttentionBlock, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads
        self.dropout_rate = dropout_rate
        assert self.head_dim * num_heads == self.embedding_size

        self.linear_to_qkv = nn.Linear(embedding_size, embedding_size * 3)
        self.out_linear = nn.Linear(embedding_size, embedding_size)
    def forward(self, x, mask=None, isCausal = False):
        '''
            X = (batch_size, seq_len, embedding_size)
        '''
        batch_size, seq_len, embedding_size = x.size()
        x = self.linear_to_qkv(x)
        '''
            X = (batch_size, seq_len, embedding_size * 3)
        '''
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim * 3)
        '''
            X = (batch_size, seq_len, num_heads, head_dim * 3)
        '''
        x = x.permute(0, 2, 1, 3)
        '''
            X = (batch_size, num_heads, seq_len, head_dim * 3)
        '''
        q, k, v = x.chunk(3, dim=-1)
        '''
            q = (batch_size, num_heads, seq_len, head_dim)
            k = (batch_size, num_heads, seq_len, head_dim)
            v = (batch_size, num_heads, seq_len, head_dim)
        '''
        out, _ = scaled_dot_attention(q, k, v, mask, dropout = self.dropout_rate, isCausal = isCausal)
        '''
            out = (batch_size, num_heads, seq_len, head_dim)
        '''
        out = out.permute(0, 2, 1, 3)
        '''
            out = (batch_size, seq_len, num_heads, head_dim)
        '''
        out = out.reshape(batch_size, seq_len, -1)
        '''
            out = (batch_size, seq_len, num_heads * head_dim)
        '''
        out = self.out_linear(out)
        return out


class MultiHeadCrossAttentionBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, droput_rate = None):
        super(MultiHeadCrossAttentionBlock, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads
        self.dropout_rate = droput_rate
        assert self.head_dim * num_heads == self.embedding_size

        self.linear_to_kv = nn.Linear(embedding_size, embedding_size * 2)
        self.linear_to_q = nn.Linear(embedding_size, embedding_size)
        self.out_linear = nn.Linear(embedding_size, embedding_size)

    def forward(self, x, y, mask=None):
        ####################### PREPARE INPUT FROM THE ENCODER #####################À
        '''
            X = (batch_size, seq_len, embedding_size)
        '''
        batch_size, seq_len_x, embedding_size = x.size()
        x = self.linear_to_kv(x)
        '''
            X = (batch_size, seq_len, embedding_size * 2) for the K and V ## ENCODER OUTPUT FOR THE CROSS ATTENTION
        '''
        x = x.reshape(batch_size, seq_len_x, self.num_heads, self.head_dim * 2)
        '''
            X = (batch_size, seq_len, num_heads, head_dim * 2)
        '''
        x = x.permute(0, 2, 1, 3)
        '''
            X = (batch_size, num_heads, seq_len, head_dim * 2)
        '''
        k, v = x.chunk(2, dim=-1)
        '''
            k = (batch_size, num_heads, seq_len, head_dim)
            v = (batch_size, num_heads, seq_len, head_dim)
        '''
        ################### PREPARE INPUT FROM THE DECODER
        _, seq_len_y, _ = y.size()
        y = self.linear_to_q(y)
        '''
            q = (batch_size, seq_len, embedding_size)
        '''
        y = y.reshape(batch_size, seq_len_y, self.num_heads, self.head_dim)
        '''
            q = (batch_size, seq_len, num_heads,head_dim)
        '''
        q = y.permute(0, 2, 1, 3)
        '''
            q = (batch_size, num_heads,seq_len, head_dim)
        '''
        ################# CROSS ATTENTION

        out, _ = scaled_dot_attention(q, k, v, mask, dropout = self.dropout_rate)
        '''
            out = (batch_size, num_heads, seq_len, head_dim)
        '''
        out = out.permute(0, 2, 1, 3)
        '''
            out = (batch_size, seq_len, num_heads, head_dim)
        '''
        out = out.reshape(batch_size, seq_len_y, -1)
        '''
            out = (batch_size, seq_len, num_heads * head_dim)
        '''
        out = self.out_linear(out)
        return out


def show_attention(q, k, v, out, attention):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8, 8))
    axes[0, 0].imshow(q.squeeze())
    axes[0, 0].set_title('q')
    axes[0, 0].set_xlabel('embedding dimension')
    axes[0, 0].set_ylabel('sequence lenght')

    axes[0, 1].imshow(k.squeeze())
    axes[0, 1].set_title('k')
    axes[0, 1].set_xlabel('embedding dimension')
    axes[0, 1].set_ylabel('sequence lenght')

    axes[0, 2].imshow(v.squeeze())
    axes[0, 2].set_title('v')
    axes[0, 2].set_xlabel('embedding dimension')
    axes[0, 2].set_ylabel('sequence lenght')

    axes[1, 0].imshow(attention.squeeze())
    axes[1, 0].set_title('attention matrix')
    axes[1, 0].set_xlabel('sequence lenght')
    axes[1, 0].set_ylabel('sequence lenght')

    axes[1, 1].imshow(out.squeeze())
    axes[1, 1].set_title('output values')
    axes[1, 1].set_xlabel('embedding dimension')
    axes[1, 1].set_ylabel('sequence lenght')
    axes[1, 2].set_axis_off()
    plt.tight_layout()
    plt.show()




def main():

    batch_size = 1
    seq_len = 5
    embedding_dim = 4 * 3
    num_heads = 1
    head_dim = embedding_dim // num_heads
    x = torch.randn(batch_size, num_heads, seq_len, head_dim*3)
    q, k, v = x.chunk(3, dim=-1)
    padding_mask = create_random_padding_mask(batch_size, seq_len)

    print("PADDING MASK", padding_mask)
    out, attention = scaled_dot_attention(q, k, v, mask=padding_mask)
    out = out.permute(0, 2, 1, 3)
    out = out.reshape(batch_size, seq_len, -1)
    q = q.permute(0, 2, 1, 3).reshape(batch_size, seq_len, embedding_dim)
    k = k.permute(0, 2, 1, 3).reshape(batch_size, seq_len, embedding_dim)
    v = v.permute(0, 2, 1, 3).reshape(batch_size, seq_len, embedding_dim)
    print(q.shape, k.shape, v.shape, out.shape, attention.shape)
    print("ATTENTION MATRIX", attention)
    show_attention(q, k, v, out, attention)
    mask = create_look_ahead_mask(batch_size, seq_len)

    multi_head_attention_block = MultiHeadSelfAttentionBlock(embedding_dim, num_heads=3)

    v = multi_head_attention_block(x)
    print(v)
if __name__ == "__main__":
    main()