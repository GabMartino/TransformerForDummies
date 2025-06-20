



import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, max_len = 10000):
        super(PositionalEncoding, self).__init__()
        self.embedding_size = embedding_size
        self.max_len = max_len
        '''
                  Create the steps for the denominators
              '''
        self.even_steps = torch.arange(0, self.embedding_size, 2, dtype=torch.long, requires_grad=False)
        self.odd_steps = torch.arange(1, self.embedding_size, 2, dtype=torch.long, requires_grad=False)
        '''
                  Create the denominators
              '''
        self.even_denominator = torch.pow(max_len, self.even_steps / self.embedding_size)
        self.odd_denominator = torch.pow(max_len, self.odd_steps / self.embedding_size)

    def forward(self, x):
        batch_size, seq_len = x.size()

        if seq_len > self.max_len:
            self.max_len = seq_len + 1000
            self.even_denominator = torch.pow(self.max_len, self.even_steps / self.embedding_size)
            self.odd_denominator = torch.pow(self.max_len, self.odd_steps / self.embedding_size)
        '''
            Create the steps for the positions in the sequence lenght
        '''
        positions = torch.arange(0, seq_len, 1, dtype=torch.long, device=x.device).unsqueeze(1)
        '''
            Create the encoding, unsqueezing the last dimension to allow the interleaving after
        '''
        even_encoding = torch.sin(positions/self.even_denominator.to(x.device)).unsqueeze(-1)
        odd_encoding = torch.cos(positions/self.odd_denominator.to(x.device)).unsqueeze(-1)
        '''
            Concat the two encodings reshaping for the interleaving of sin and cos values
        '''
        encoding = torch.cat((even_encoding, odd_encoding), dim=-1).reshape(seq_len, -1)#.view(-1, even_encoding.shape[-1])
        return encoding




def show_positional_encoding(encoding):
    from matplotlib import colors
    encoding = encoding.numpy()

    fig, ax = plt.subplots()
    ax.imshow(encoding, vmax=1, vmin=-1)

    plt.title("Positional Encoding")
    plt.xlabel("Position in the Embedding Lenght")
    plt.ylabel("Position in the Sequence Lenght")
    plt.show()

def main():
    batch_size = 1
    seq_len = 100
    embedding_size = 256
    x = torch.randn((batch_size, seq_len))
    encoding = PositionalEncoding(embedding_size)(x)
    print(encoding.shape)
    show_positional_encoding(encoding)
if __name__ == '__main__':
    main()