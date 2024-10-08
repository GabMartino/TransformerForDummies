


import torch
import torch.nn as nn

from model.blocks.LayerNormalization import LayerNormalization
from model.blocks.MultiHeadAttentionBlock import MultiHeadCrossAttentionBlock, MultiHeadSelfAttentionBlock


class DecoderLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, ff_hidden_size, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads
        assert self.head_dim * num_heads == self.embedding_size

        self.masked_multihead_attention = MultiHeadSelfAttentionBlock(embedding_size, num_heads)
        self.layer_norm_1 = LayerNormalization(self.embedding_size)

        self.multihead_cross_attention = MultiHeadCrossAttentionBlock(embedding_size, num_heads)

        self.ff_model = nn.Sequential(nn.Linear(embedding_size, ff_hidden_size),
                                      nn.ReLU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(ff_hidden_size, embedding_size),
                                      nn.Dropout(dropout))
        self.layer_norm_2 = LayerNormalization(self.embedding_size)

    def forward(self, x, decoder_mask, encoder_output):
        residual = x
        '''
            NOTE:
            -> THE MASK HERE FOR THE SELF ATTENTION BLOCK OF THE DECODER IS
                MASK = LOOK AHEAD MASK + PADDING MASK 
        '''
        x = self.masked_multihead_attention(x, mask=decoder_mask)
        x = self.layer_norm_1(x + residual)


        residual = x
        x = self.multihead_cross_attention(encoder_output, x, mask=None) ##check mask
        x = self.layer_norm_2(x + residual)

        residual = x
        x = self.ff_model(x)
        x = self.layer_norm_2(x + residual)
        return x




class Decoder(nn.Module):
    def __init__(self, embedding_size, num_heads, ff_hidden_size, dropout, num_layers):
        super(Decoder, self).__init__()

        self.decoder_layers = nn.ModuleList([DecoderLayer(embedding_size, num_heads, ff_hidden_size, dropout) for _ in range(num_layers)])

    def forward(self, x, decoder_mask, encoder_output):
        '''
            The encoder output remains the same for all decoder layers
        '''
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, decoder_mask, encoder_output)
        return x

def main():
    batch_size = 10
    seq_len = 100
    embedding_size = 376
    x = torch.randn((batch_size, seq_len, embedding_size))

    decoder_layer = DecoderLayer(embedding_size, num_heads=8, ff_hidden_size=2048, dropout=0.1)
    encoder_output = torch.randn((batch_size, seq_len, embedding_size))
    x = decoder_layer(x, None, encoder_output)

    decoder = Decoder(embedding_size, num_heads=8, ff_hidden_size=2048, dropout=0.1, num_layers=5)
    x = decoder(x, None, encoder_output)
    print(x.shape)
if __name__ == '__main__':
    main()

