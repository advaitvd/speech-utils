import torchaudio.functional as F
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
from tqdm import tqdm
import audiosegment
import os
import json


def get_token_offsets(emission, transcript, waveform, dictionary, sample_rate=16000):

    def align(emission, tokens):
        targets = torch.tensor([tokens], dtype=torch.int32)
        alignments, scores = F.forced_align(emission, targets, blank=0)

        alignments, scores = alignments[0], scores[0]
        scores = scores.exp()
        return alignments, scores

    def unflatten(list_, lengths):
        assert len(list_) == sum(lengths)
        i = 0
        ret = []
        for l in lengths:
            ret.append(list_[i : i + l])
            i += l
        return ret

    def compute_alignments(emission, transcript, dictionary):
        tokens = []
        for word in transcript:
            for char in word:
                try:
                    tokens.append(dictionary[char])  
                except KeyError:
                    tokens.append(dictionary.get('<unk>', -1))  
                    print(f"Character '{char}' not found in dictionary. Using <unk>")
        alignment, scores = align(emission, tokens)
        token_spans = F.merge_tokens(alignment, scores)
        word_spans = unflatten(token_spans, [len(word) for word in transcript])
        return word_spans

    def _score(spans):
        return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)


    word_spans = compute_alignments(emission, transcript, dictionary)
    num_frames = emission.shape[1]
    ratio = waveform.shape[0] / num_frames

    token_offsets = []
    for word, spans in zip(transcript, word_spans):
        x0 = int(ratio * spans[0].start)
        x1 = int(ratio * spans[-1].end)
        start = x0 / sample_rate
        end = x1 / sample_rate
        token_offsets.append({
            'token': word.lower(),
            'start_offset': round(start, 3),
            'end_offset': round(end, 3),
        })

    return token_offsets


def load_model(model="Harveenchadha/vakyansh-wav2vec2-marathi-mrm-100"):
    w2v_processor = Wav2Vec2Processor.from_pretrained(model)
    w2v_model = Wav2Vec2ForCTC.from_pretrained(model)
    lang_dict = w2v_processor.tokenizer.get_vocab()
    return w2v_model, w2v_processor, lang_dict


def main(audio_file, transcript, model="Harveenchadha/vakyansh-wav2vec2-marathi-mrm-100", out_dir='./out'):
    os.makedirs(out_dir, exist_ok=True)
    model, processor, dictionary = load_model(model=model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    transcript = open(transcript, 'r').read().strip().replace('\n', '$ ')
    if transcript[-2:] != '$ ':
        transcript += '$'

    TRANSCRIPT = transcript.split()

    waveform, sample_rate = librosa.load(audio_file, sr=16_000)

    emissions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(waveform), 30 * sample_rate), desc="[Computing Emissions]", total=len(waveform)// (30 * sample_rate)):
            chunk = waveform[i:i + 30 * sample_rate]
            input_values = processor(chunk, sampling_rate=sample_rate, return_tensors="pt").to(device)
            emission = model(**input_values).logits.to('cpu')
            emissions.append(emission)

    emission = torch.cat(emissions, dim=1)

    token_offsets = get_token_offsets(emission, TRANSCRIPT, waveform, dictionary, sample_rate=sample_rate)
    print(json.dumps(token_offsets, indent=2))
    start_sent = None
    audio = audiosegment.from_file(audio_file)

    chunk_idx = 0
    for token_span in token_offsets:
        tok = token_span['token']
        start = int(token_span['start_offset'] * 1000)
        end = int(token_span['end_offset'] * 1000)
        if not start_sent:
            start_sent = start

        if tok[-1] == '$':
            end_sent = end + 1
            audio[start_sent:end_sent].export(os.path.join(out_dir, os.path.basename(audio_file.replace('.wav', f'_{chunk_idx:05d}.wav'))))
            chunk_idx += 1
            start_sent = None



if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, required=True)
    parser.add_argument('--transcript', type=str, required=True)
    parser.add_argument('--model', type=str, default="Harveenchadha/vakyansh-wav2vec2-marathi-mrm-100")
    parser.add_argument('--out', type=str, default="./out")
    args = parser.parse_args()
    main(args.audio, args.transcript, args.model, args.out)
