import json

from nltk.tokenize import word_tokenize
from tqdm import tqdm


def main():


    English_Vocab = {}
    with open("../Datasets/it-en/europarl-v7.it-en.en", "r") as f:
        for line in tqdm(f):
            v = line.strip()
            for token in word_tokenize(v):
                English_Vocab[str(token).lower()] = True

    for idx, k in tqdm(enumerate(English_Vocab.keys())):
        English_Vocab[k] = idx

    with open("./english_vocab.json", "w") as f:
        json.dump(English_Vocab, f)

    ############################################################################à
    Italian_vocab = {}
    with open("../Datasets/it-en/europarl-v7.it-en.it", "r") as f:
        for line in tqdm(f):
            v = line.strip()
            for token in word_tokenize(v):
                Italian_vocab[str(token).lower()] = True

    for idx, k in enumerate(Italian_vocab.keys()):
        Italian_vocab[k] = idx
    print(len(Italian_vocab.keys()))
    with open("./italian_vocab.json", "w") as f:
        json.dump(Italian_vocab, f)
if __name__ == "__main__":
    main()