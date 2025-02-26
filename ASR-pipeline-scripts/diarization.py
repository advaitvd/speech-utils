import torch
from pyannote.audio import Pipeline
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
import audiosegment
import os
import audiosegment
import librosa
import soundfile as sf
from functools import cache

TOKEN='SET TOKEN HERE'
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=TOKEN)
pipeline.to(torch.device(device))

@cache
def load_vad_pipeline(model, token, min_duration_on, min_duration_off, device):
    model = Model.from_pretrained(
        model,
        token=token,
        use_auth_token=token
    )
    pipeline = VoiceActivityDetection(segmentation=model)
    HYPER_PARAMETERS = {
        "min_duration_on": min_duration_on,
        "min_duration_off": min_duration_off
    }
    pipeline.instantiate(HYPER_PARAMETERS)
    return pipeline.to(device)


# Hard boundry chunking - lagacy
def split_audio_v0(file_path, chunk_length_ms, output_dir, main_audio_filename):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()[1:]
    audio, sr = librosa.load(file_path, sr=16000)
    os.makedirs(output_dir, exist_ok=True)

    chunk_length_samples = int(chunk_length_ms * (sr / 1000))

    for i, start in enumerate(range(0, len(audio), chunk_length_samples)):
        chunk = audio[start:start + chunk_length_samples]
        chunk_name = os.path.join(output_dir, f"{main_audio_filename}_chunk_{i + 1}.wav")
        sf.write(chunk_name, chunk, sr)
        print(f"Exported {chunk_name}")


# VAD boundary chunking - advait.dhopeshwarkar@tihiitb.org
def split_audio_v1(filepath, chunk_length_ms, output_dir, main_audio_filename):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    pipeline = load_vad_pipeline("pyannote/segmentation-3.0",
                                 token=TOKEN,
                                 min_duration_on=0,
                                 min_duration_off=0.2,
                                 device=device)

    vad = pipeline(filepath)

    audio, sr = librosa.load(filepath, sr=16000)
    os.makedirs(output_dir, exist_ok=True)

    start = 0
    idx = 0
    for item in vad._tracks:
        end = int(item.end * 1000)
        # if chunk length is greater than 70% of chunk length parameter, consider spliting at this point.
        if end - start >= (chunk_length_ms * 0.7):
            chunk = audio[int(start * sr / 1000) : int(end * sr / 1000)]
            chunk_name = os.path.join(output_dir, f'{main_audio_filename}_chunk_{idx + 1}.wav')
            sf.write(chunk_name, chunk, sr)
            idx += 1
            start = end
    
    #r remaining audio
    if end > start:
        chunk = audio[int(start * sr / 1000) : int(end * sr / 1000)]
        chunk_name = os.path.join(output_dir, f'{main_audio_filename}_chunk_{idx + 1}.wav')
        sf.write(chunk_name, chunk, sr)
        print(f"Exported {chunk_name}")
    

def split_audio(*args, **kwargs):
    # Change version as per requirements
    return split_audio_v1(*args, **kwargs)


def diarize_audio(filename, num_speakers=None):
    if num_speakers:
        diarization = pipeline(filename, num_speakers=num_speakers)
    else:
        diarization = pipeline(filename)

    actual_result = {}
    for segment in diarization.itersegments():
        speaker_id = diarization.get_labels(segment).pop()
        start, end = segment
        actual_result[(start, end)] = speaker_id

    # print("actual_result: ", actual_result)
    speaker_list = list(actual_result.values())
    updated_speaker_list = swap_elements_by_first_occurrence(speaker_list)
    for idx, key in enumerate(actual_result.keys()):
        actual_result[key] = updated_speaker_list[idx]

    return actual_result


def retrieve_chunks(timestamps, start_time, end_time):
    chunks = []
    for (start, end), speaker in timestamps.items():
        if start_time >= start and end_time <= end:
            chunks.append((round(start_time, 3), round(end_time, 3), speaker))
        elif start_time <= start and end_time >= end:
            chunks.append((round(start, 3), round(end, 3), speaker))
        elif start_time <= start < end_time or start_time < end <= end_time:
            chunks.append((round(max(start_time, start), 3), round(min(end_time, end), 3), speaker))
    return chunks


def swap_all(lst, old_value, new_value):
    swapped_idx = []
    for i, value in enumerate(lst):
        if value == old_value:
            lst[i] = new_value
            swapped_idx.append(i)
        if value == new_value and i not in swapped_idx:
            lst[i] = old_value
            swapped_idx.append(i)
    return lst

def swap_elements_by_first_occurrence(input_list):
    first_occurrence = {}
    my_list_swapped = None
    for i, speaker in enumerate(input_list):
        if speaker not in first_occurrence:
            first_occurrence[speaker] = i

    sorted_keys = sorted(first_occurrence.keys(), key=lambda x: int(x.split('_')[-1]))

    new_index = {sorted_keys[index]: value for index, (key, value) in enumerate(first_occurrence.items())}

    swap_history = {}
    for k, v in new_index.items():
        old_value = input_list[v]
        new_value = k
        if swap_history.get(new_value, None) == old_value:
            continue
        my_list_swapped = swap_all(input_list, old_value, new_value)
        swap_history[old_value] = new_value

    return my_list_swapped
