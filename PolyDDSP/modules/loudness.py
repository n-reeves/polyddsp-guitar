"""
Extractor function for loudness envelopes
"""

import torch
import torch.nn as nn
import torchaudio


class LoudnessExtractor(nn.Module):
    """
    Loudness envelope extractor

    Args:
        sr: input audio sample rate
        frame_length: length of each frame window
        attenuate_gain: gain multiplier applied at end of generation
        pitch_extractor: changes padding method to agree with basic pitch module
        device: Specify whether computed on cpu, cuda or mps

    Input: Audio input of size (batch, samples)
    Output: Loudness envelopes of size (batch, frames)
    """

    def __init__(
        self,
        sr: int = 16000,
        frame_length: int = 256,
        attenuate_gain: float = 2.0,
        device: str = "mps",
    ):
        super(LoudnessExtractor, self).__init__()

        self.sr = sr
        self.frame_length = frame_length
        self.n_fft = frame_length * 8
        self.device = device
        self.attenuate_gain = attenuate_gain
        self.smoothing_window = nn.Parameter(
            torch.hann_window(self.n_fft, dtype=torch.float32), requires_grad=False
        )

        self.to(device)

    def A_weighting(self, frequencies: torch.Tensor, min_db: int = -45) -> torch.Tensor:
        """
        Calculate A-weighting in Decibel scale
        mirrors the librosa function of the same name

        Args:
            frequencies: tensor of frequencies to return weight
            min_db: minimum decibel weight to avoid exp/log errors

        Returns: Decibel weights for each frequency bin
        """

        f_sq = frequencies**2.0
        const = (
            torch.tensor(
                [12194.217, 20.598997, 107.65265, 737.86223],
                dtype=torch.float32,
                device=self.device,
            )
            ** 2.0
        )

        weights = 2.0 + 20.0 * (
            torch.log10(const[0])
            + 2 * torch.log10(f_sq)
            - torch.log10(f_sq + const[0])
            - torch.log10(f_sq + const[1])
            - 0.5 * torch.log10(f_sq + const[2])
            - 0.5 * torch.log10(f_sq + const[3])
        )

        if min_db is None:
            return weights
        else:
            return torch.maximum(
                torch.tensor([min_db], dtype=torch.float32).to(self.device), weights
            )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute loudness envelopes for audio input
        """

        # padded_audio = F.pad(audio, (self.n_fft // 2, self.n_fft // 2))
        # sliced_audio = padded_audio.unfold(1, self.n_fft, self.frame_length)
        # sliced_windowed_audio = sliced_audio * self.smoothing_window

        # compute FFT step
        
        #basic pitch computes metrics by adding padding to the front of the audio
        
        s = torch.stft(
            audio,
            hop_length=self.frame_length ,
            n_fft=self.n_fft,
            window=self.smoothing_window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
            )

        amplitude = torch.abs(s)
        power = amplitude**2

        frequencies = torch.fft.rfftfreq(self.n_fft, 1 / self.sr, device=self.device)
        a_weighting = self.A_weighting(frequencies).unsqueeze(0).unsqueeze(0)

        weighting = 10 ** (a_weighting / 10)
        # print(weighting)
        # print(power)
        power = power.transpose(-1, -2) * weighting

        avg_power = torch.mean(power, -1)
        loudness = torchaudio.functional.amplitude_to_DB(
            avg_power, multiplier=10, amin=1e-4, db_multiplier=10
        )
        
        return loudness
