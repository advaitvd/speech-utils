# audio_extractor.py

from moviepy import VideoFileClip
import librosa
import os
import soundfile as sf
from diarization import split_audio

def extract_audio(video_path, audio_output_path):
    # Load the video file
    video = VideoFileClip(video_path)
    
    audio = video.audio
    
    audio.write_audiofile(audio_output_path, codec='pcm_s16le')

def process_video_audio(video_input, audio_dir, chunk_length_s, lang, split, diarization_func):
    chunk_length_ms = chunk_length_s * 1000
    lang_audio_dir = os.path.join(audio_dir, lang)
    os.makedirs(lang_audio_dir, exist_ok=True)

    def process_file(file_path):
        main_filename = os.path.splitext(os.path.basename(file_path))[0]
        if file_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            audio_output_path = os.path.join(lang_audio_dir, main_filename + '.wav')
            extract_audio(file_path, audio_output_path)
        else:
            audio_output_path = file_path
        
        # Perform diarization on the whole audio
        diarization_result = diarization_func(audio_output_path)
        
        if split:
            chunk_dir = os.path.join(lang_audio_dir, main_filename + '_chunks')
            split_audio(audio_output_path, chunk_length_ms, chunk_dir, main_filename)
        
        return diarization_result

    diarization_results = {}

    if os.path.isdir(video_input):
        video_files = [f for f in os.listdir(video_input) if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wav', '.mp3', '.ogg'))]
        for video_file in video_files:
            video_path = os.path.join(video_input, video_file)
            try:
                diarization_results[video_file.split('.')[0]] = process_file(video_path)
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
                continue

    elif os.path.isfile(video_input):
        diarization_results[os.path.basename(video_input)] = process_file(video_input)
    else:
        raise ValueError("The provided video_input path is neither a directory nor a file.")
    
    return diarization_results

# Example usage
# video_input = "/path/to/videos"
# audio_dir = "/path/to/audios"
# lang = "hindi"
# chunk_length_s = 120  # 120 seconds (2 minutes)
# split = True
# diarization_results = process_video_audio(video_input, audio_dir, chunk_length_s, lang, split, diarize_audio)
