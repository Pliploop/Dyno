import torchaudio
import soundfile as sf
import numpy as np
import torch
import librosa

def load_audio_chunk(path, target_n_samples, target_sr, start = None, verbose = False):
    # info = sf.info(path)
    # frames = info.frames
    # sr = info.samplerate
    
    info = torchaudio.info(path, backend='soundfile')
    sr = info.sample_rate
    frames = info.num_frames
    
    print(f'length of audio in seconds: {frames/sr}') if verbose else None
    print(f'Original sample rate: {sr}') if verbose else None
    
    
    if path.split('.')[-1] == 'mp3':
        frames = frames - 8192
    
    new_target_n_samples = int(target_n_samples * sr / target_sr)
    
    print(f'New target n samples: {new_target_n_samples}') if verbose else None
    
    if start is None:
        # random
        start = np.random.randint(0, frames - new_target_n_samples)
        
    # audio,sr = sf.read(path, start=start, stop=start+new_target_n_samples, always_2d=True, dtype='float32')
    audio,sr = torchaudio.load(path, frame_offset=start, num_frames=new_target_n_samples, backend='soundfile')
    # audio = torch.tensor(audio.T)
    # resample to target sample rate
    
    # print(audio.shape)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
        print(f'Resampled to {target_sr}, shape of audio: {audio.shape}') if verbose else None
    
    # print(audio.shape)    
    return audio

def load_full_audio(path, target_sr, verbose = False):
    try:
        audio, sr = sf.read(path, always_2d=True, dtype='float32')
    except:
        audio, sr = librosa.load(path, sr=None)
    # audio, sr = torchaudio.load(path, backend='soundfile')
    # if the audio file is stereo, mean that dimension inplace
    
    print(f'length of audio in seconds: {audio.shape[0]/sr}') if verbose else None
    
    audio = torch.tensor(audio.T)
    if audio.shape[0] == 2:
        audio = audio.mean(dim=0, keepdim=True)
    # resample to target sample rate
    print(f'Original sample rate: {sr}, shape of audio: {audio.shape}') if verbose else None
    audio = torchaudio.functional.resample(audio, sr, target_sr) if sr != target_sr else audio
    print(f'Resampled to {target_sr}, shape of audio: {audio.shape}') if verbose else None
    return audio


def load_full_and_split(path, target_sr, target_n_samples, hop = None, verbose = False):
    hop = target_n_samples if hop is None else hop
    audio = load_full_audio(path, target_sr, verbose=verbose)
    audio = audio.squeeze()
    #if audio is shorter than target_n_samples, repeat until it is
    if audio.shape[0] < target_n_samples:
        n_repeats = int(np.ceil(target_n_samples / audio.shape[0]))
        audio = audio.repeat(n_repeats)
    
    audio = audio.unfold(0, int(target_n_samples), int(hop)).unsqueeze(1)
    # print(audio.shape)
    print(f'Split audio into {audio.shape[0]} chunks') if verbose else None
    return audio