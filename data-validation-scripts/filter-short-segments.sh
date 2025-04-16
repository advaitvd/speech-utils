#!/usr/bin/bash

# Author : Advait Vinay Dhopeshwarkar (advait.dhopeshwarkar@tihiitb.org)
# Date   : 15-04-2025

# Run this script before ASR inference to filter out short utterances from the data.
# Use the wav.filtered.scp as the scp file as input for the asr.py script.

export PATH=$PATH:/workspace/advait/bin/

segments_dir_path=$1

scp_orig=$segments_dir_path/wav.scp
text_orig=$segments_dir_path/text
utt2dur_dump=$segments_dir_path/utt2dur

utt2dur $scp_orig > $utt2dur_dump
filter-dur --min_dur 0.5 --utt2dur $utt2dur_dump --scp $scp_orig
filter-dur --min_dur 0.5 --utt2dur $utt2dur_dump --scp $text_orig
