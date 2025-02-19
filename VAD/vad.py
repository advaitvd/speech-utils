from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
import fire
import torch
import audiosegment
import os
from tqdm import tqdm

def parse_file(file):
    with open(file) as f:
        lines = f.read().strip().split('\n')
        utt2wav = {line.split()[0]: line.split()[1] for line in lines}
    return utt2wav

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

def main(
    scp_file,
    output_dir,
    token='AUTH_TOKEN',
    min_duration_on=0.0,
    min_duration_off=1.0,
    debug_mode=False
    ):

    buffer = min_duration_off / 2 # buffer to add before annd after the speech segment in seconds => half of min_duration_off to keep natural flow of speech

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    _, pipeline = load_vad_pipeline("pyannote/segmentation-3.0", token)
    pipeline = initialize_pipeline(pipeline, min_duration_on, min_duration_off, device)

    utt2wav = parse_file(scp_file)

    output_audio = os.path.join(output_dir, 'audio')
    os.makedirs(output_audio, exist_ok=True)

    out_scp = os.path.join(output_dir, 'wav.scp')
    stats_file = os.path.join(output_dir, 'stats.txt')
    
    scp_f = open(out_scp, 'w')
    stats_f = open(stats_file, 'w')

    for utt, wav in tqdm(utt2wav.items(), desc='Processing', total=len(utt2wav)):
        vad = pipeline(wav)
        audio = audiosegment.from_file(wav)
        speech_out = []
        silence_out = []
        segments_info = []
        silence_start = 0
        for item in vad._tracks:
            start = max(0, int(item.start * 1000) - int(buffer * 1000))
            end = min(int(item.end * 1000) + int(buffer * 1000), len(audio))

            if end < start:
                continue
            
            speech_out.append(audio[start:end])
            silence_out.append(audio[silence_start:start])

            if debug_mode:
                if start > 0:
                    audio[silence_start:start].export(os.path.join(output_audio, f"{utt}_{silence_start}_{start}_NS.wav"), format='wav')
                audio[start:end].export(os.path.join(output_audio, f"{utt}_{start}_{end}_S.wav"), format='wav')
            

            segments_info.append((silence_start / 1000, start / 1000, end / 1000))
            silence_start = end
    
        silence_out.append(audio[silence_start:])

        speech_o = speech_out[0]
        for a in speech_out[1:]:
            speech_o += a

        silence_o = silence_out[0]
        for a in silence_out[1:]:
            silence_o += a
        

        out_path = os.path.join(output_audio, utt+'.wav')
        speech_o.export(out_path, format='wav')
        silence_o.export(out_path.replace('.wav', '.silence.wav'), format='wav')

        with open(out_path.replace('.wav', '.segments.text'), 'w') as f:
            for silence_start, start, end in segments_info:
                f.write(f"NS :\t{silence_start}-->{start}\n")
                f.write(f"S  :\t{start}-->{end}\n")

        scp_f.write(f"{utt}\t{out_path}\n")
        stats_f.write(f"{utt}\tinput={len(audio)/1000:0.2f}s\toutput={len(speech_o)/1000:0.2f}\treduction={len(silence_o) / len(audio) * 100:0.2f}%\n")

    scp_f.close()
    stats_f.close()


if __name__=="__main__":
    fire.Fire(main)
