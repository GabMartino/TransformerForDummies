import json

from nltk.tokenize import word_tokenize
from tqdm import tqdm


def main():
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers

    # Initialize a tokenizer with BPE model
    tokenizer = Tokenizer(models.BPE())

    # Pre-tokenize by whitespace
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Trainer to build vocab
    trainer = trainers.BpeTrainer(vocab_size=15000, special_tokens=["[PAD]", "[SOS]", "[EOS]"])

    # Load and train on your corpus
    files = ["../Datasets/it-en/europarl-v7.it-en.it"]
    tokenizer.train(files, trainer)

    # Save tokenizer
    tokenizer.save("bpe_italian_vocab.json")

    # Encode text
    output = tokenizer.encode("Example for Hugging Face BPE tokenizer.")
    print(output.tokens)
if __name__ == "__main__":
    main()