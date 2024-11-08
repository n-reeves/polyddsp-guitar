import random
import os
import librosa
import matplotlib.pyplot as plt

import torch
import torchaudio

from PolyDDSP.modules.loudness import LoudnessExtractor

def spec_plot(audio, 
              sr: int=22050, 
              n_fft: int =512,
              hop_length: int =128,
              save_png: bool = False, 
              png_name: str='test.png'):
    """
    Spec plot function from Deep Learning for Audio Module
    
    Args:
        audio: (1,N) numpy
        sr: sample rate of audio
        n_fft: n fft used in stft
        hop_length: hop length in samples
        save_png: whether or not spectrogram png is saved
        png_name: name of the png file
    """
    X = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S = librosa.amplitude_to_db(abs(X)) 
    
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='jet')
    plt.colorbar(format='%+2.0f dB')
    
    if save_png:
        plt.savefig(png_name)
        
    plt.show()

def load_audio(filepath: str, 
             sr: int = 22050):
    """
    utility function for loading audio at a sample rate
    
    Args:
        filepath: path to an audio file
        sr: sample rate 
        
    Out:
        torch tensor containing audio file at sample rate
    """
    audio, nat_sr = torchaudio.load(filepath) #dim: ch x samples 
    if sr is not None and sr != nat_sr :
        audio = torchaudio.functional.resample(audio, nat_sr, sr)
    return audio


def crop_audio(audio: torch.tensor,
               segment_length: int = 44100,
              random_crop: bool = True):
    """
    utility function for loading audio at a sample rate
    
    Args:
        audio: tensor containing audio sample (1,samples)
        segment_length: number of samples in the cropped audio clip
        random_crop: bool indicating whether or not to randomly select start position of cropped segment
        
    Out:
        torch tensor containing cropped audio file
    """
    audio_length = audio.shape[-1]
    
    if audio_length < segment_length:
        pad_len = segment_length - audio_length
        pad = torch.zeros((1,pad_len))
        audio = torch.cat((audio,pad),dim=1)
        
    if random_crop:
        start_ind = random.randint(0, audio_length - segment_length)
        end_ind = start_ind + segment_length
        cropped_audio = audio[:,start_ind:end_ind]
        
    else:
        start_ind = 0
        end_ind = segment_length
        cropped_audio = audio[:,:segment_length]
    
    return {'audio':cropped_audio, 'start_ind':start_ind, 'end_ind':end_ind}


def train_test_split(audio_dir: str, 
                train_pct: float = .80, 
                valid_pct: float =.1,
                seed: int = 1):
    """
    returns train, test, and validation sets given a directory of audio files
    
    Inputs:
        audio_dir: path to audio files
        train_pct: percent of the audio files delegated to train set
        valid_pct: percent of the audio files delegated to the validation set
        
    out: dictionary of lists with file paths 
        
    """
    random.seed(seed)
    audio_files = os.listdir(audio_dir)
    audio_files_clean = [x for x in audio_files if x.endswith('.wav')]
    
    random.shuffle(audio_files_clean)
    
    train_count = int(len(audio_files_clean) * train_pct)
    valid_count = int(len(audio_files_clean) * valid_pct)
    
    train_set = audio_files_clean[:train_count]
    valid_set = audio_files_clean[train_count:train_count+valid_count]
    test_set = audio_files_clean[train_count+valid_count:]
    
    out = {'train':train_set, 'valid':valid_set, 'test':test_set}
    return out


# Code taken directly from object in PolyDDSP pitch.py
def normalised_to_db(audio: torch.Tensor):
        """
        Convert spectrogram to dB and normalise

        Args:
            audio: The spectrogram input. (batch, 1, freq_bins, time_frames)
                or (batch, freq_bins, time_frames)

        Returns:
            the spectogram in dB in the same shape as the input
        """
        power = torch.square(audio)
        log_power = 10.0 * torch.log10(power + 1e-10)

        log_power_min = torch.min(log_power, keepdim=True, dim=-2)[0]
        log_power_min = torch.min(log_power_min, keepdim=True, dim=-1)[0]
        log_power_offset = log_power - log_power_min
        log_power_offset_max = torch.max(log_power_offset, keepdim=True, dim=-2)[0]
        log_power_offset_max = torch.max(log_power_offset_max, keepdim=True, dim=-1)[0]

        log_power_normalised = log_power_offset / log_power_offset_max
        return log_power_normalised
    

#exp sigmoid from PolyDDSP
#bounds output between 0 and 2. used because it improved training stability in monophonic DDSP models
def modified_sigmoid(x):
    x = x.sigmoid()
    x = x.pow(2.3026) 
    x = x.mul(2.0)
    x.add_(1e-7)
    return x


def get_mean_std_loudness(audio_dir, file_list, sr = 22050):
    """
    Calculates

    Args:
        audio_dir: directory of audio files 
        file_list: list of audio file names to calculate dataset metrics for
        sr: sample rate used for loudness extraction. Audio is resampled to this value if native sample rate is different

    Returns:
        dictionary containing mean and standard deviation of loudness across the files in file_list
    """
    
    loudness_extractor = LoudnessExtractor(sr=sr, device = 'cpu')
    
    all_loud = []
    
    for file in file_list:
        audio_path = os.path.join(audio_dir, file)
        audio = load_audio(audio_path, sr=sr)
        
        loudness = loudness_extractor(audio)
        all_loud.append(loudness)
    
    all_loud_cat = torch.cat(all_loud, dim=-1)
    
    loud_mean = torch.mean(all_loud_cat)
    loud_std = torch.std(all_loud_cat)
    
    return {'mean':loud_mean.item(), 'std':loud_std.item()}
        