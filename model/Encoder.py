import torch
import torch.nn as nn

from model.blocks.LayerNormalization import LayerNormalization
from model.blocks.MultiHeadAttentionBlock import MultiHeadSelfAttentionBlock


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
                                      nn.Linear(ff_hidden_size, embedding_size),
                                      nn.Dropout(dropout))
        self.layer_norm_2 = LayerNormalization(embedding_size)

    def forward(self, x, mask = None):
        residual = x
        x = self.multi_head_attention(x, mask=mask)
        x = self.layer_norm_1(x + residual)
        residual = x
        x = self.ff_model(x)
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

    x = torch.randn(batch_size, seq_len, embedding_size)
    '''
        Create a random padding mask
    '''
    padding_start_indeces = torch.randint(1, seq_len, (batch_size, 1)).squeeze()
    padding_mask = torch.full((batch_size, seq_len), False, dtype=torch.bool)
    for i in range(padding_start_indeces.shape[0]):
        padding_mask[i, padding_start_indeces[i]:] = True

    padding_mask = padding_mask.float()
    padding_mask[padding_mask == 1.] = -torch.inf
    print(padding_mask)
    encoder = Encoder(embedding_size, num_heads, ff_hidden_size, dropout=0.1, num_encoder_layers=1)
    x = encoder(x, mask = padding_mask)
    print(x.shape)



if __name__ == '__main__':
    main()