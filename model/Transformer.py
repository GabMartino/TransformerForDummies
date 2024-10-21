from multiprocessing.managers import Value

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
        self.target_embedding = nn.Embedding(config.decoder.vocab_size,
                                             config.embedding_size)
        self.decoder = Decoder(config.embedding_size,
                               config.num_heads,
                               config.decoder.ff_hidden_size, \
                               config.dropout,
                               config.decoder.num_layers)
        '''
            For both 
        '''
        self.target_embedding = SharedWeightsEmbedding(config.decoder.vocab_size, config.embedding_size)

        if config.same_source_target_vocabulary:
            self.source_embedding = self.target_embedding
        else:
            self.source_embedding = nn.Embedding(config.encoder.vocab_size,
                                                     config.embedding_size)

        self.positional_encoding = PositionalEncoding(config.embedding_size)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, encoder_input,  decoder_input, source_mask_keys = None, target_mask_keys = None, cross_attention_mask_keys=None):
        source_embedding = self.source_embedding(encoder_input)
        '''
            The output is of shape (batch_size, seq_len, embedding_size)
        '''
        source_encoding = self.positional_encoding(encoder_input)
        '''
           Add positional encoding to the embedding
       '''
        source_embedding = source_embedding + source_encoding
        source_embedding = self.dropout(source_embedding)

        '''
            Create source mask from the encoder keys
        '''
        source_mask = None
        if source_mask_keys is not None:
            source_mask_right = source_mask_keys.unsqueeze(1).repeat(1, encoder_input.shape[1], 1)
            source_mask_left = source_mask_right.transpose(-1, -2)
            source_mask = (source_mask_left | source_mask_right).float()
            source_mask[source_mask == 1.] = -torch.inf

        '''
            ENCODER
        '''
        encoder_output = self.encoder(x=source_embedding,
                                      mask=source_mask)
        '''
            Prepare Decoder's input
        '''
        target_embedding = self.target_embedding(decoder_input)
        target_encoding = self.positional_encoding(decoder_input)
        target_embedding = target_embedding + target_encoding
        target_embedding = self.dropout(target_embedding)

        '''
            Prepare decoder's input mask
        '''
        target_mask = None
        if target_mask_keys is not None:
            target_mask_right = target_mask_keys.unsqueeze(1).repeat(1, encoder_input.shape[1], 1)
            target_mask_left = target_mask_right.transpose(-1, -2)
            target_mask = (target_mask_left | target_mask_right).float()
            target_mask[target_mask == 1.0] = -torch.inf

        '''
            Prepare cross attention mask
        '''
        cross_attention_mask = None
        if cross_attention_mask_keys is not None:
            if cross_attention_mask_keys == "encoder":
                source_mask_right = source_mask_keys.unsqueeze(1).repeat(1, encoder_input.shape[1], 1)
                cross_attention_mask = source_mask_right.float()
                cross_attention_mask[cross_attention_mask == 1.] = -torch.inf
            else:
                raise ValueError("The cross attention mask should be either 'encoder' or a new mask")

        '''
            DECODER
        '''
        out = self.decoder(x=target_embedding, decoder_mask=target_mask,
                           encoder_output=encoder_output, cross_attention_mask=cross_attention_mask)
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
