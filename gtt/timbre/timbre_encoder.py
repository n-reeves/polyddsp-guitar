import torchaudio
import torch
from torch import nn

class TimbreEncoder(nn.Module):
    """
        network based on design in Engel et al., 2020 DDSP paper
    
    Args:
        sr: sample rate of audio
        n_mfcc: number of mfcc features
        n_mels: number of mel bands
        n_ftt: number of fft bins
        hop_length: hop length in samples
        gru_features: number of features in gru layer
        gru_bidirectional: whether or not gru is bidirectional
        timbre_enc_size: size of timbre encoding
    """
    def __init__(self,
                 sr: int = 22050,
                 n_mfcc: int = 30,
                 n_mels: int = 128,
                 n_ftt: int = 1024,
                 hop_length: int = 256,
                 gru_features: int = 512,
                 gru_bidirectional: bool = True,
                 timbre_enc_size: int = 16
                 
                ):
        super().__init__()
        
        self.n_mfcc = n_mfcc
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs=dict(
                n_fft=n_ftt,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=20.0,
                f_max=8000.0,)
        )
        self.norm = nn.InstanceNorm1d(n_mfcc, affine=True)
        
        self.gru = nn.GRU(
            input_size=n_mfcc,
            hidden_size=gru_features,
            num_layers=1,
            batch_first=True,
            bidirectional=gru_bidirectional,
        )
        
        linear_in_features = 2*gru_features if gru_bidirectional else gru_features
        self.linear_out = nn.Linear(in_features = linear_in_features,
                                   out_features = timbre_enc_size)
        
    def forward(self, x):
        """
        Input: torch tensor of shape (batch, samples)
        Output: 
            torch tensor of shape (batch, mfcc frames, timbre_enc_size) containg timbre latents
        """
        
        x = self.mfcc(x) #(batch, n_mfcc, mfcc frames)
        x = self.norm(x) #(batch, n_mfcc, mfcc frames)
        x = x.permute(0, 2, 1) #(batch, mfcc frames, n_mfcc)
        x = self.gru(x)[0] #(batch, mfcc frames, gru features(*2 if bidirectional))
        x = self.linear_out(x) #(batch, mfcc frames, timbre_enc_size)
        
        return x