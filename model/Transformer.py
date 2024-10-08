import torch
import torch.nn as nn
from confection import Config

from model.Decoder import Decoder
from model.Encoder import Encoder
from model.blocks.PositionalEncoding import PositionalEncoding
from model.utils.utils import create_random_padding_mask, create_look_ahead_mask


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        '''
            For the encoder...
        '''
        self.source_embedding = nn.Embedding(config.encoder.vocab_size,
                                             config.embedding_size)
        self.encoder = Encoder(config.embedding_size,
                               config.num_heads,
                               config.encoder.ff_hidden_size, \
                               config.dropout,
                               config.encoder.num_layers)

        '''
            For the decoder... 
        '''
        self.target_embedding = nn.Embedding(config.decoder.vocab_size,
                                             config.embedding_size)
        self.decoder = Decoder(config.embedding_size,
                               config.num_heads,
                               config.decoder.ff_hidden_size, \
                               config.dropout,
                               config.decoder.num_layers)
        self.out_linear = nn.Linear(config.embedding_size,
                                    config.decoder.vocab_size)
        '''
            For both 
        '''
        self.positional_encoding = PositionalEncoding(config.embedding_size)

    def forward(self, x,  y, source_mask = None,target_mask = None):
        source_embedding = self.source_embedding(x)
        '''
            The output is of shape (batch_size, seq_len, embedding_size)
        '''
        source_encoding = self.positional_encoding(x)
        '''
           Add positional encoding to the embedding
       '''
        source_embedding = source_embedding + source_encoding
        '''
            ENCODER
        '''
        encoder_output = self.encoder(source_embedding, source_mask)
        '''
            DECODER
        '''
        target_embedding = self.target_embedding(y)
        target_encoding = self.positional_encoding(y)
        target_embedding = target_embedding + target_encoding
        out = self.decoder(x=target_embedding, decoder_mask=target_mask, encoder_output=encoder_output)
        '''
            OUTPUT
        '''
        out = self.out_linear(out)
        return out




def main():

    vocab_size = 10000
    batch_size = 16
    seq_len = 128

    config = Config()
    config.encoder.vocab_size = vocab_size
    config.encoder.num_layers = 1
    config.encoder.ff_hidden_size = 512

    config.decoder.vocab_size = vocab_size
    config.decoder.num_layers = 1
    config.decoder.ff_hidden_size = 512

    config.embedding_size = 512
    config.dropout = 0.1
    config.num_heads = 8

    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    source_padding_mask = create_random_padding_mask(batch_size, seq_len)
    target_padding_mask = create_random_padding_mask(batch_size, seq_len)
    look_ahead_mask = create_look_ahead_mask(batch_size, seq_len)

    transformer = Transformer(config)
    x = transformer(x, source_mask=source_padding_mask, target_mask=target_padding_mask)
    print(x.shape)
if __name__ == "__main__":
    main()
