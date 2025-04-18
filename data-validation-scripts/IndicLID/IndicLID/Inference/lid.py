from ai4bharat.IndicLID import IndicLID
import argparse
import os
from tqdm import tqdm
import logging

# Author : Advait Vinay Dhopeshwarkar (advait.dhopeshwarkar@tihiitb.org)
# Date   : 15-04-2025

# Downgrade the transformers version you are using to 4.37.0
# Latest version of transformers library cause issue with the bert model
# object that is loaded using the IndicLID library. (The checkpoint was
# likely created using older version of transformers.)
# `pip install transformers==4.37.0` Should work fine.

parser = argparse.ArgumentParser()
parser.add_argument("--text", required=True)
args = parser.parse_args()

text = args.text
output = os.path.join(os.path.dirname(text), "utt2lid")
logfile = os.path.join(os.path.dirname(text), "lid.log")

logging.basicConfig(
            filename=logfile,
            filemode='w',
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d-%m-%Y %H:%M:%S"
        )

with open(text, 'r') as f:
    lines = [line.split('\t') for line in f.read().strip().split('\n')]
    utt2text = {utt:text for utt, text in lines}

# The model object is created with parameters that the authors of IndicLID
# use in their Demo.
model = IndicLID(input_threshold=0.5, roman_lid_threshold=0.6)
utt2lid = {}
batch_size = 1
progbar = tqdm(total=len(utt2text), desc="PROGRESS")

for utt, text in utt2text.items():
    try:
        lid = model.batch_predict([text], batch_size)[0][1]
        utt2lid[utt] = lid
        logging.info(f"{utt} Successful!")
    except Exception as e:
        logging.info(f"{utt} Failed! Exception('{e}'). Skipping...")
        raise Exception(e)

    progbar.update(1)

with open(output, 'w') as f:
    for utt, lid in utt2lid.items():
        f.write(f"{utt}\t{lid}\n")

