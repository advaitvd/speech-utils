import argparse
import json
from collections import defaultdict
import re
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_text", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--vocab_size", default=1024, type=int)
    return parser.parse_args()

def train(text, vocab_size=1024):
    vocab = set([c for c in text])
    vocab.add('_')

    words = ["_" + word for word in text.split(' ')]
    word2freq = defaultdict(int)
    for word in words:
        word2freq[word] += 1

    wordsplitfreq = [([c for c in word], freq) for word, freq in word2freq.items()]
    merges = []
    while len(vocab) < vocab_size:
        # compute combination frequencies for current combined words in train text
        combination_freqs = defaultdict(int)
        for word_list, freq in wordsplitfreq:
            for a, b in zip(word_list[:-1], word_list[1:]):
                combination_freqs[(a, b)] += freq
        
        if len(combination_freqs) == 0:
            # cannot combine any further
            break
        
        (a, b), freq = max(list(combination_freqs.items()), key=lambda x: x[-1])

        # add merge rule and add the merged to vocab
        merges.append((a, b))
        vocab.add(a + b)
        # merge all a, b combinations
        wordsplitfreq_updated = []
        
        for word_list, freq in wordsplitfreq:
            word_list_new, i = [], 0
            last_added = False
            while i < len(word_list) - 1:
                x, y = word_list[i], word_list[i + 1]
                if x == "_tokenizatio" and a == "_tokenizatio":
                    import pdb
                    pdb.set_trace()
                if (x, y) == (a, b):
                    word_list_new.append(x + y)
                    i += 2
                    if i == len(word_list):
                        last_added = True
                    else:
                        y = word_list[i]
                else:
                    word_list_new.append(x)
                    i += 1
            if not last_added:
                word_list_new.append(y)
            wordsplitfreq_updated.append((word_list_new, freq))
        wordsplitfreq = wordsplitfreq_updated

    save_dict = {
        "merge_rules": merges,
        "vocab": list(vocab)
    }
    print(len(vocab))
    return save_dict

class Tokenizer:
    def __init__(self, merge_rules, vocab, xos="<sos/eos>"):
        self.merge_rules = merge_rules
        self.xos  = xos
        vocab.append(xos)
        self.vocab = vocab
        self.subword2int = {subword: i for i, subword in enumerate(vocab)}

    def tokenize(self, text):
        word_list = ["_" + word for word in text.split(' ')]
        tokens = [c for word in word_list for c in word]

        for a, b in self.merge_rules:
            tokens_merged, i = [], 0
            last_added = False
            while i < len(tokens) - 1:
                x, y = tokens[i], tokens[i + 1]
                if (x, y) == (a, b):
                    tokens_merged.append(x + y)
                    i += 2
                    if i == len(tokens):
                        last_added = True
                    else:
                        y = tokens[i]
                else:
                    tokens_merged.append(x)
                    i += 1
            if not last_added:
                tokens_merged.append(y)
            tokens = tokens_merged
        return [self.xos] + tokens + [self.xos]

    def token2int(self, tokens):
        return [self.subword2int(tok) for tok in tokens]


def load_model(model_json):
    with open(model_json, 'r') as f:
        model = json.load(f)
    return model


def main(args):
    train_text = args.train_text
    save_dir = args.save_dir
    vocab_size = args.vocab_size
    
    os.makedirs(save_dir, exist_ok=True)
    model_save = os.path.join(save_dir, "bpemodel.json")

    with open(train_text, 'r') as f:
        text = f.read().split("\n")
        text = ' '.join(text)
        text, _ = re.subn(r"\s", ' ', text)
    
    model = train(text, vocab_size)

    with open(model_save, 'w') as f:
        json.dump(model, f)
    
    tokenizer = Tokenizer(
        merge_rules=model["merge_rules"],
        vocab=model["vocab"],
    )

    print(tokenizer.tokenize("Will you watch the spaceX launch next weekend?"))


if __name__ == "__main__":
    args = get_args()
    main(args)