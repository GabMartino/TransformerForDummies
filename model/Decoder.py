


import torch
import torch.nn as nn

from model.blocks.LayerNormalization import LayerNormalization
from model.blocks.MultiHeadAttentionBlock import MultiHeadCrossAttentionBlock, MultiHeadSelfAttentionBlock
from model.utils.utils import create_look_ahead_mask, create_random_padding_mask


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
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, encoder_output, decoder_padding_mask = None, out_encoder_mask_keys=None):
        residual = x
        '''
            NOTE:
            -> THE MASK HERE FOR THE SELF ATTENTION BLOCK OF THE DECODER IS
                MASK = LOOK AHEAD MASK + PADDING MASK 
        '''
        x = self.masked_multihead_attention(x, mask=decoder_padding_mask, isCausal = True)
        x = self.dropout(x)
        x = self.layer_norm_1(x + residual)

        residual = x
        x = self.multihead_cross_attention(encoder_output, x, mask=out_encoder_mask_keys) ##check mask
        x = self.dropout(x)
        x = self.layer_norm_2(x + residual)

        residual = x
        x = self.ff_model(x)
        x = self.dropout(x)
        x = self.layer_norm_2(x + residual)
        return x




class Decoder(nn.Module):
    def __init__(self, embedding_size, num_heads, ff_hidden_size, dropout, num_layers):
        super(Decoder, self).__init__()

        self.decoder_layers = nn.ModuleList([DecoderLayer(embedding_size, num_heads, ff_hidden_size, dropout) for _ in range(num_layers)])

    def forward(self, x, decoder_padding_mask, encoder_output, out_encoder_mask_keys=None):
        '''
            The encoder output remains the same for all decoder layers
        '''
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x=x,
                              decoder_padding_mask=decoder_padding_mask,
                              encoder_output=encoder_output, out_encoder_mask_keys=out_encoder_mask_keys)
        return x

def main():
    batch_size = 1
    seq_len = 10
    embedding_size = 376


    x = torch.randn((batch_size, seq_len, embedding_size))
    mask = create_random_padding_mask(batch_size, seq_len)
    print(mask)
    '''
        Test single decoder layer
    '''
    decoder_layer = DecoderLayer(embedding_size, num_heads=8, ff_hidden_size=2048, dropout=0.1)
    encoder_output = torch.randn((batch_size, seq_len, embedding_size)) ## Simulate encoder output
    x = decoder_layer(x, mask, encoder_output)
    '''
        Test multi layer decoder 
    '''
    decoder = Decoder(embedding_size, num_heads=8, ff_hidden_size=2048, dropout=0.1, num_layers=5)
    x = decoder(x, mask, encoder_output)
    '''
        Test decoder layer with a lookahead mask
    '''
    causal_mask = create_look_ahead_mask(batch_size, seq_len)
    print(causal_mask)
    encoder_output = torch.randn((batch_size, seq_len, embedding_size))  ## Simulate encoder output
    x = decoder_layer(x=x, decoder_mask=causal_mask, encoder_output=encoder_output)
if __name__ == '__main__':
    main()

