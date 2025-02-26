import time
import os
from audio_extractor import process_video_audio
from force_alignment import transcribe_audio
from xml_creation import create_transcript_xml
from xml_merger import merge_xml_files, update_merged_xml_with_diarization
from diarization import diarize_audio
import shutil
import librosa

def check_len_audio(audio_path, threshold=0.05):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        return duration > threshold
    except Exception as e:
        print(f"Error checking audio length: {e}")
        return False
    
def organize_solo_file(video_path):
    base_path = os.path.dirname(video_path)
    input_dir_dir = os.path.join(base_path, "input_dir")
    os.makedirs(input_dir_dir, exist_ok=True)    
    destination_path = os.path.join(input_dir_dir, os.path.basename(video_path))
    shutil.copy(video_path, destination_path)    
    print(f"Copied {video_path} to {destination_path}")
    return input_dir_dir

def run_pipeline(input_dir, audio_dir, lang, json_dir, transcript_xml_dir, model, chunk_length_s=120, split=True):
    diarization_results = process_video_audio(input_dir, audio_dir, chunk_length_s, lang, split, diarize_audio)
    print(diarization_results)
    
    lang_audio_dir = os.path.join(audio_dir, lang)
    print("-" * 60)
    print(os.walk(lang_audio_dir))
    
    for root, _, files in os.walk(lang_audio_dir):
        for file in files:
            if file.endswith('.wav') and ('chunk' in file or not split):
                chunk_audio_path = os.path.join(root, file)
                if check_len_audio(chunk_audio_path):
                    word_offset = transcribe_audio(chunk_audio_path, lang.capitalize(), json_dir, model)
                    
                    if len(word_offset) > 0:
                        main_audio_filename = os.path.basename(os.path.dirname(chunk_audio_path))
                        chunk_number = int(file.split('_chunk_')[-1].split('.')[0])
                        main_audio_xml_dir = os.path.join(transcript_xml_dir, lang, main_audio_filename)
                        create_transcript_xml(word_offset, {}, main_audio_filename, chunk_number, main_audio_xml_dir, lang)
                    
    for main_audio_filename in os.listdir(os.path.join(transcript_xml_dir, lang)):
        if os.path.isdir(os.path.join(transcript_xml_dir, lang, main_audio_filename)):
            mrg_main_audio_xml_dir = os.path.join(transcript_xml_dir, lang, main_audio_filename)
            output_file = os.path.join(transcript_xml_dir, lang, f"{main_audio_filename}_merged.xml")
            merge_xml_files(mrg_main_audio_xml_dir, output_file, lang, chunk_length_s)

            getter = main_audio_filename.split("_chunks")[0]
            diarization_result = diarization_results.get(getter, {})
            print(getter, diarization_result)
            update_merged_xml_with_diarization(output_file, diarization_result)

input_dir = "/workspace/advait/workspace/asr-pipeline/Audios"
audio_dir = "/workspace/advait/workspace/asr-pipeline/Chunks"
os.makedirs(audio_dir, exist_ok=True)

lang = "Hindi"  # Allowed languages: Hindi, Telugu, Marathi
json_dir = "/workspace/advait/workspace/asr-pipeline/JSON"
transcript_xml_dir = "/workspace/advait/workspace/asr-pipeline/XML"

os.makedirs(json_dir, exist_ok=True)
os.makedirs(transcript_xml_dir, exist_ok=True)

chunk_length_s = 20
model = 'indicconformer' # Allowed models : indicconformer, wav2vec2, whisper

start_time = time.time()

if not os.path.isdir(input_dir):
    input_dir = organize_solo_file(input_dir)
    run_pipeline(input_dir, audio_dir, lang, json_dir, transcript_xml_dir, model, chunk_length_s, split=True)
    end_time = time.time()
    transcription_time = end_time - start_time
    print(f"Time taken to transcribe the directory: {transcription_time:.2f} seconds")
else:
    run_pipeline(input_dir, audio_dir, lang, json_dir, transcript_xml_dir, model, chunk_length_s, split=True)
    end_time = time.time()
    transcription_time = end_time - start_time
    print(f"Time taken to transcribe the directory: {transcription_time:.2f} seconds")
