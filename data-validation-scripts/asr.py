import torch
import soundfile as sf
import nemo.collections.asr as nemo_asr
import argparse
import os
import logging
from tqdm import tqdm
import subprocess

# Author : Advait Vinay Dhopeshwarkar (advait.dhopeshwarkar@tihiitb.org)
# Date   : 14-04-2025

def convert(wav):
    os.makedirs("tmp/", exist_ok=True)
    subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", wav,
                "-ac", "1",
                "-ar", "16000",
                "tmp/temp.wav"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="wav.scp file path containing paths to input audios")
    parser.add_argument("--model", required=True)
    parser.add_argument("--lang_id", required=True)
    args = parser.parse_args()

    model = args.model
    scp = args.input
    lang_id = args.lang_id
    output = os.path.join(os.path.dirname(scp), "hyp")
    logfile = os.path.join(os.path.dirname(scp), "asr.log")

    logging.basicConfig(
            filename=logfile,
            filemode='w',
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
    )

    with open(scp, 'r') as f:
        utt2wav = [utt.split('\t') for utt in f.read().strip().split('\n')]
        utt2wav = {utt:wav for utt, wav in utt2wav}

    model = nemo_asr.models.EncDecCTCModel.restore_from(model)
    model.eval()
    model = model.to(device)
    model.cur_decoder = "ctc"
    out_writer = open(output, 'w')
    progbar = tqdm(total=len(utt2wav), desc="PROCESSED")
    for utt, wav in utt2wav.items():
        convert(wav)
        ctc_text = model.transcribe(["tmp/temp.wav"], batch_size=1, language_id=lang_id, return_hypotheses=True)[0]
        score = ctc_text[0].score
        text = ctc_text[0].text
        out_writer.write(f"{utt}\t{text}\t{score}\n")
        logging.info(f"{utt} - Inference finished successfully!")
        progbar.update(1)

    out_writer.close()

if __name__=="__main__":
    main()

