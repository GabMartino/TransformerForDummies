import torch
import torch.nn as nn

from model.blocks.LayerNormalization import LayerNormalization
from model.blocks.MultiHeadAttentionBlock import MultiHeadSelfAttentionBlock
from model.utils.utils import create_random_padding_mask


class EncoderLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, ff_hidden_size, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads
        assert self.head_dim * num_heads == self.embedding_size

        self.multi_head_attention = MultiHeadSelfAttentionBlock(embedding_size, num_heads)
        self.layer_norm_1 = LayerNormalization(embedding_size)
        self.ff_model = nn.Sequential(nn.Linear(embedding_size, ff_hidden_size),
                                      nn.ReLU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(ff_hidden_size, embedding_size))
        self.layer_norm_2 = LayerNormalization(embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        residual = x
        x = self.multi_head_attention(x, mask=mask)
        x = self.dropout(x)
        x = self.layer_norm_1(x + residual)
        residual = x
        x = self.ff_model(x)
        x = self.dropout(x)
        x = self.layer_norm_2(x + residual)
        return x


class Encoder(nn.Module):
    def __init__(self, embedding_size, num_heads, ff_hidden_size, dropout, num_encoder_layers):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList([EncoderLayer(embedding_size, num_heads, ff_hidden_size, dropout) for _ in range(num_encoder_layers)])

    def forward(self, x, mask = None):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        return x



def main():
    batch_size = 10
    seq_len = 15
    embedding_size = 128
    num_heads = 8
    ff_hidden_size = 1024
    '''
        Create a random input for the Encoder
    '''
    x = torch.randn(batch_size, seq_len, embedding_size)
    '''
        Create a random padding mask
    '''
    padding_mask = create_random_padding_mask(batch_size, seq_len)
    print(padding_mask)
    encoder = Encoder(embedding_size, num_heads, ff_hidden_size, dropout=0.1, num_encoder_layers=4)
    x = encoder(x, mask = padding_mask)
    print(x.shape)



if __name__ == '__main__':
    main()