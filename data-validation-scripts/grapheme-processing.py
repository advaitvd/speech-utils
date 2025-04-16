import re
import os
import argparse
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--log", required=True)
    return parser.parse_args()

def main():
    args = get_args()
    text = args.text
    log = args.log

    with open(text, 'r') as f:
        lines = [line.split('\t') for line in f.read().strip().split('\n')]
        utt2text = {utt:text for utt, text in lines}

    log_f = open(log, 'w')

    vocab_stats = {}
    grapheme_stats = {}

    pattern1 = r"\[[^\]]*\]"
    pattern2 = r"<[^>]*>"
    pattern3 = r'[?\.,";:\(\)!{}#]'
    pattern4 = r'[_\-\~=]'

    for utt, text in utt2text.items():
        # Remove the tags to get the grapheme information.
        text, _ = re.subn(pattern1, '', text)
        text, _ = re.subn(pattern2, '', text)
        text, _ = re.subn(pattern3, '', text)
        text, _ = re.subn(pattern4, ' ', text)
        for word in text.split(' '):
            if word == '':
                continue
            # VOCAB STATS
            vocab_stats[word] = vocab_stats.get(word, 0) + 1
            
            # GRAPHEME STATS
            for c in word:
                grapheme_stats[c] = grapheme_stats.get(c, 0) + 1

    stats = {
            'vocab': vocab_stats,
            'grapheme': grapheme_stats,
    }
    
    json.dump(stats, log_f, indent=2)
    log_f.close()

if __name__=="__main__":
    main()
