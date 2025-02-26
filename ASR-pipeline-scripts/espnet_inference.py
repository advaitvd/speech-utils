from espnet2.bin.asr_inference import Speech2Text
import torch
import time
import librosa
import soundfile as sf


class EspnetInference:
    def __init__(
        self,
        asr_train_config='/workspace/advait/workspace/asr-pipeline/artefacts/exp/config.yaml',
        asr_model_file='/workspace/advait/workspace/asr-pipeline/artefacts/exp/20epoch.pth',
        beam_size=4
        ):
        '''
        asr_train_config: path to the asr_train.yaml file that is generated in the exp/ directory
        asr_model_file: path to the checkpoint file
        beam_size: beam size for beam search decoding
        '''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.speech2text = Speech2Text(
            asr_train_config=asr_train_config,
            asr_model_file=asr_model_file,
            device=self.device,
            beam_size=beam_size
        )
    
    @classmethod
    def load_audio(self, audio):
        wav, rate = sf.read(audio)
        if rate != 16000:
            wav = librosa.resample(wav, rate, 16000)
            rate = 16000
        return wav, rate

    def __call__(self, audio: str):
        wav, rate = self.load_audio(audio)
        transcript = self.speech2text(wav)[0][0].replace('।', '').strip()
        return transcript
