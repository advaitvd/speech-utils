from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
import torch
import audiosegment
import os

# Author : Advait Vinay Dhopeshwarkar (advait.dhopeshwarkar@tihiitb.org)
# Date   : 14-04-2025


def load_vad_pipeline(model, token):
    model = Model.from_pretrained(
        model,
        token=token,
        use_auth_token=token
    )
    pipeline = VoiceActivityDetection(segmentation=model)
    return model, pipeline

def initialize_pipeline(pipeline, min_duration_on, min_duration_off, device):
    HYPER_PARAMETERS = {
        "min_duration_on": min_duration_on,
        "min_duration_off": min_duration_off
    }
    pipeline.instantiate(HYPER_PARAMETERS)
    return pipeline.to(device)

class SilenceSpeechCheck:
    def __init__(self, token, min_duration_on=0.0, min_duration_off=1.0):
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.buffer = min_duration_off / 2

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        _, pipeline = load_vad_pipeline("pyannote/segmentation-3.0", token)
        self.pipeline = initialize_pipeline(pipeline, min_duration_on, min_duration_off, device)
    
    def __call__(self, audio_file):
        audio = audiosegment.from_file(audio_file)
        if len(audio) == 0:
            return 0, 0, 0

        vad = self.pipeline(audio_file)

        speech_length = 0

        for item in vad._tracks:
            start = max(0, int(item.start * 1000) - int(self.buffer * 1000))
            end = min(int(item.end * 1000) + int(self.buffer * 1000), len(audio))
            speech_length += (end - start)
        
        silence_length = len(audio) - speech_length
        speech_length /= 1000
        silence_length /= 1000
        # speech, silence, total
        return speech_length, silence_length, speech_length + silence_length

def get_hms(time):
    tot_minutes = time // 60
    hours = int(tot_minutes // 60)
    minutes = int(tot_minutes % 60)
    seconds = int(time % 60)
    return hours, minutes, seconds


def main():
    import sys
    from tqdm import tqdm
    scp = sys.argv[1]
    logfile = sys.argv[2]
    log_f = open(logfile, 'w')
    
    with open(scp, 'r') as f:
        lines = [line.split() for line in f.read().strip().split('\n')]

    utt2wav = {utt:wav for utt, wav in lines}

    token = ""
    check = SilenceSpeechCheck(token)
    tot_speech = 0.0
    tot_audio = 0.0
    for utt, wav in tqdm(utt2wav.items(), total=len(utt2wav), desc="Processed"):
        speech_len, silence_len, audio_len = check(wav)
        log_f.write(f"{utt}\tSpeech: {speech_len:.2f}\tSilence: {silence_len:.2f}\tTotal: {audio_len:0.2f}\n")
        tot_speech += speech_len
        tot_audio += audio_len

    speech_h, speech_m, speech_s = get_hms(tot_speech)
    audio_h, audio_m, audio_s = get_hms(tot_audio)

    print(f"Total Speech Duration: {tot_speech:0.2f}s ({speech_h:02}:{speech_m:02}:{speech_s:02})")
    print(f"Total Audio Duration: {tot_audio:0.2f}s ({audio_h:02}:{audio_m:02}:{audio_s:02})")


if __name__ == "__main__":
    main()
