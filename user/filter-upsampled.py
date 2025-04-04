import torchaudio
import numpy as np


def is_upsampled_from_8k(audio_path, threshold_ratio=0.05):
    waveform, sr = torchaudio.load(audio_path) 
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        sr = 16000
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    hop_length = int(0.010 * sr)  # 10 ms
    n_fft = 1024

    spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)(waveform)
    
    sound = spec[0].log2().numpy() > -10
    n, m = sound.shape
    total_sound = sound.sum().item() / n / m * 100
    upper_band_sound = sound[(n // 2) + 1:, :].sum().item() / (n - n // 2) / m * 100
    return upper_band_sound / total_sound < threshold_ratio


def is_upsampled_from_8k_v2(audio_path, threshold_ratio=0.02):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        sr = 16000

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    spectrum = np.abs(np.fft.rfft(waveform.numpy()[0]))
    freqs = np.fft.rfftfreq(len(waveform[0]), 1 / sr)

    total_energy = np.sum(spectrum)

    high_freq_band = (freqs >= 4000) & (freqs <= 8000)
    high_freq_energy = np.sum(spectrum[high_freq_band])

    ratio = high_freq_energy / total_energy
    return (ratio < threshold_ratio).item()


def main():
    import sys
    from tqdm import tqdm
    import logging

    scp = sys.argv[1]
    if len(sys.argv) == 2:
        print("Log file not provided. Defaulting to filter-upsampled.log")
    logfile = "filter-upsampled.log" if len(sys.argv) == 2 else sys.argv[2]

    logging.basicConfig(
        filename=logfile,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    with open(scp, 'r') as f:
        utts = [utt.split('\t') for utt in f.read().strip().split('\n')]
        utt2wav = {utt:wav for utt, wav in utts}

    for utt, wav in tqdm(utt2wav.items(), total=len(utt2wav), desc="[PROCESSING]"):
        if is_upsampled_from_8k_v2(wav):
            logging.info(f"{utt} might be upsampled from 8kHz. Please check the spectrogram for {wav}")
        else:
            logging.info(f"{utt} Ok!")


if __name__=="__main__":
    main()

