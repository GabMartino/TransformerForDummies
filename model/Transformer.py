
import torch
import torch.nn as nn
from confection import Config
from model.Decoder import Decoder
from model.Encoder import Encoder
from model.blocks.PositionalEncoding import PositionalEncoding
from model.blocks.SharedWeightsEmbedding import SharedWeightsEmbedding
from model.utils.utils import create_random_padding_mask, create_look_ahead_mask


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        '''
            For the encoder...
        '''
        self.encoder = Encoder(config.embedding_size,
                               config.num_heads,
                               config.encoder.ff_hidden_size, \
                               config.dropout,
                               config.encoder.num_layers)

        '''
            For the decoder... 
        '''
        self.target_embedding = nn.Embedding(num_embeddings=config.decoder.vocab_size,
                                             embedding_dim=config.embedding_size,
                                             padding_idx=config.decoder.padding_idx)
        self.decoder = Decoder(config.embedding_size,
                               config.num_heads,
                               config.decoder.ff_hidden_size, \
                               config.dropout,
                               config.decoder.num_layers)
        '''
            For both 
        '''
        self.target_embedding = SharedWeightsEmbedding(vocab_size=config.decoder.vocab_size,
                                                       embedding_size=config.embedding_size,
                                                       padding_idx=config.decoder.padding_idx)

        if config.same_source_target_vocabulary:
            self.source_embedding = self.target_embedding
        else:
            self.source_embedding = nn.Embedding(num_embeddings=config.encoder.vocab_size,
                                                     embedding_dim=config.embedding_size,
                                                 padding_idx=config.encoder.padding_idx)

        self.positional_encoding = PositionalEncoding(config.embedding_size)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, encoder_input,  decoder_input, source_padding_mask_keys = None, target_padding_mask_keys = None, out_encoder_mask_keys=None):
        '''
            ENCODER
        '''
        '''
            After this step the size will be:
            (batch_size, L, embedding_size)
        '''
        source_embedding = self.source_embedding(encoder_input)
        source_embedding = source_embedding + self.positional_encoding(encoder_input)
        source_embedding = self.dropout(source_embedding)
        encoder_output = self.encoder(x=source_embedding,
                                      padding_mask=source_padding_mask_keys)

        '''
        
            DECODER
        '''
        target_embedding = self.target_embedding(decoder_input)
        target_embedding = target_embedding + self.positional_encoding(decoder_input)
        target_embedding = self.dropout(target_embedding)
        if out_encoder_mask_keys is None:
            out_encoder_mask_keys = source_padding_mask_keys
        out = self.decoder(x=target_embedding, decoder_padding_mask=target_padding_mask_keys,
                           encoder_output=encoder_output, out_encoder_mask_keys=out_encoder_mask_keys)
        '''
            OUTPUT
        '''
        out = self.target_embedding.inverse_forward(out)
        return out




def main():

    vocab_size = 10000
    batch_size = 1
    seq_len = 10

    config = Config()
    config.encoder = Config()
    config.encoder.vocab_size = vocab_size
    config.encoder.num_layers = 1
    config.encoder.ff_hidden_size = 512

    config.decoder = Config()
    config.decoder.vocab_size = vocab_size
    config.decoder.num_layers = 1
    config.decoder.ff_hidden_size = 512

    config.embedding_size = 512
    config.dropout = 0.1
    config.num_heads = 8

    config.same_source_target_vocabulary = False

    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    source_padding_mask = create_random_padding_mask(batch_size, seq_len)
    target_padding_mask = create_random_padding_mask(batch_size, seq_len)
    look_ahead_mask = create_look_ahead_mask(batch_size, seq_len)

    transformer = Transformer(config)
    y = torch.randint(0, vocab_size, (batch_size, seq_len))
    x = transformer(x, y=y, source_mask=source_padding_mask, target_mask=target_padding_mask)
    print(x.shape)
if __name__ == "__main__":
    main()
