import sys
import re

text = sys.argv[1]
language = sys.argv[2]

with open(text, 'r') as f:
    lines = [line.split('\t') for line in f.read().strip().split('\n')]
    utt2text = {utt:text for utt, text in lines}

remove_pattern1 = re.compile(r'\[[a-zA-Z0-9\-$_ ]+\]')
remove_pattern2 = re.compile(r'<lang:Foreign>[a-zA-Z0-9 ]+</lang:Foreign>')

language_script_pattern = {
        'marathi': re.compile(r'[\u0900-\u097F]+'),
        'hindi': re.compile(r'[\u0900-\u097F]+'),
        'assamese': re.compile(r'[\u0980-\u09FF]+'),
        'bengali': re.compile(r'[\u0980-\u09FF]+'),
        'gujarati': re.compile(r'[\u0A80-\u0AFF]+'),
        'kannada': re.compile(r'[\u0C80-\u0CFF]'),
        'maithili': re.compile(r'[\u0900-\u097F]+'),
        'dogri': re.compile(r'[\u11800-\u1184F]'),
        'tamil': re.compile(r'[\u0B80-\u0BFF]'),
        'malayalam': re.compile(r'[\u0D00-\u0D7F]+'),
        'oriya': re.compile(r'[\u0B00-\u0B7F]+'),
        'punjabi': re.compile(r'[\u0A00-\u0A7F]+'),
        'telugu': re.compile(r'[\u0C00-\u0C7F]+'),
        }[language]

total = 0
count_without_match = 0
for utt, text in utt2text.items():
    match = language_script_pattern.search(text)
    if not match:
        print(f"{utt}\t{text}")
        count_without_match += 1
    total += 1

print(f"{count_without_match} out of {total} utterances don't have any language specific characters.")
