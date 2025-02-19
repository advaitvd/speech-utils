# Steps to generate AUTH_TOKEN.
# Login into huggingface. If you don't have an account make one.
# Go to 'https://huggingface.co/settings/tokens' and create a token.
# Give it appropriate name, scroll down and add the 'pyannote/segmentation-3.0' repo to repositories permissionn
# Go to 'https://huggingface.co/pyannote/segmentation-3.0' and add the org details for using the model.

export CUDA_VISIBLE_DEVICES=0 # Set the appropriate device here

python vad.py \
       /workspace/advait/workspace/VAD/test-audio/wav_5min.scp \
       ./test-working \
       --token TOKEN_HERE

# input scp in the format:
# <utt_id>    <path_to_audio_file>
# eg: 0001    /path/to/audio/file.wav
