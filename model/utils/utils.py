

import torch


def create_random_padding_mask(batch_size, seq_len):
    '''
       Create a random padding mask
    '''
    padding_start_indeces = torch.randint(1, seq_len, (batch_size, 1)).squeeze()
    if batch_size == 1:
        padding_start_indeces = padding_start_indeces.unsqueeze(0)
    padding_mask = torch.full((batch_size, seq_len), False, dtype=torch.bool)
    for i in range(batch_size):
        padding_mask[i, padding_start_indeces[i]:] = True

    padding_mask_right = padding_mask.repeat(1, seq_len, 1)
    padding_mask_left = padding_mask_right.transpose(-1, -2)
    padding_mask = (padding_mask_left | padding_mask_right).float()
    padding_mask[padding_mask == 1.] = -torch.inf
    return padding_mask


def create_look_ahead_mask(batch_size, seq_len):
    mask = torch.full((seq_len, seq_len), fill_value=-torch.inf)
    mask = torch.triu(mask, diagonal=1).unsqueeze(0)
    mask = mask.repeat(batch_size, 1, 1)
    return mask

def token_to_text(tokenized_sentence, vocabulary):
    inv_map = {v: k for k, v in vocabulary.items()}
    sentence = []
    for token in tokenized_sentence:
        word = inv_map.get(token, "[UNK]")
        sentence.append(word)
    return sentence


def main():
    batch_size = 1
    seq_len = 5
    padding_mask = create_random_padding_mask(batch_size, seq_len)
    print(padding_mask)

    look_ahead_mask = create_look_ahead_mask(batch_size, seq_len)
    print(look_ahead_mask)

    combined_mask = look_ahead_mask + padding_mask
    print(combined_mask)
if __name__ == '__main__':
    main()