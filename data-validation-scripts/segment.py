import sys
import json
import os
from pydub import AudioSegment
from tqdm import tqdm

# Author : Advait Vinay Dhopeshwarkar (advait.dhopeshwarkar@tihiitb.org)
# Date   : 14-04-2025

root = sys.argv[1]
segments = os.path.join(root, 'Segments')
os.makedirs(segments, exist_ok=True)

scp = os.path.join(root, "wav.filtered.scp")

with open(scp, 'r') as f:
    lines = [line.split() for line in f.read().strip().split('\n')]
    utt2wav = {utt:wav for utt, wav in lines}

def make_segments(transcript, wav, segments):
    uid = wav.split('/')[-1].replace('.wav', '')
    
    segment_dir_path = os.path.join(segments, uid)
    audio_dump_path = os.path.join(segment_dir_path, 'Audios')
    os.makedirs(audio_dump_path, exist_ok=True)
    
    scp_path = os.path.join(segment_dir_path, 'wav.scp')
    text_path = os.path.join(segment_dir_path, 'text')

    scp_writer = open(scp_path, 'w')
    text_writer = open(text_path, 'w')


    with open(transcript, 'r') as f:
        transcript_dict = json.load(f)

    audio = AudioSegment.from_file(wav)


    for segment in transcript_dict.get('segments', []):
        if segment.get('segmentType', '') != 'Speech':
            continue

        segment_id = segment['segmentId']

        segment_id = f"{uid}_{segment_id}"
        start = int(segment['start'] * 1000)
        end = int(segment['end'] * 1000)
        audio_seg = audio[start:end]
        segment_wav_path = os.path.join(audio_dump_path, segment_id + '.wav')
        audio_seg.export(segment_wav_path, format='wav')
        scp_writer.write(f"{segment_id}\t{segment_wav_path}\n")
        text_writer.write(f"{segment_id}\t{segment.get('transcription', None)}\n")

    scp_writer.close()
    text_writer.close()


for utt, wav in tqdm(utt2wav.items(), total=len(utt2wav), desc="Segmented"):
    transcript = wav.replace('Audio', 'Transcription').replace('.wav', '.json')
    if os.path.exists(transcript):
        make_segments(transcript, wav, segments)
