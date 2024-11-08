import torch
from torch import nn
import torchaudio

from gtt.decoder import GttDecoder
from gtt.timbre.timbre_encoder import TimbreEncoder

from PolyDDSP.modules.synth import AdditiveSynth
from PolyDDSP.modules.noise import FilteredNoise
from PolyDDSP.modules.reverb import Reverb

#old network architecture. kept for reference and documentation is not updated

class GttNet(nn.Module):
    """
    Args:
        sr: sample rate of input audio
        synth_unit_frames: frames synthesized per input feature
        device: model device
        attenuate_gain: 
        harmonic_n_controls: the number of controls produced by decoder corresponding 
            to number of harmonics for each F0
        harmonic_norm_below_nyquist:
        harmonic_amp_resample: 
        harmonic_angular_cumsum:
        noise_filters: number of controls produced by decoder used to control filtered noise synth
        noise_initial_bias: 
        noise_window_size:
        reverb_trainable: whether or not reverb is trained to learn room
        reverb_length_seconds: Length of reverb applied to summation of harmonic and noise synths
        reverb_add_dry: 
        reverb_ir: whether or not impluse response is used in reverb
    
    Return:
        (batch, samples)
    
    """
    def __init__(
        self,
        sr: int = 22050,
        input_length_seconds: int = 4,
        synth_unit_frames: int = 64,
        device: str = "cpu",
        #decoder parameters
        mlp_hidden_features: int = 512,
        mlp_blocks: int = 3,
        gru_bidirectional: bool = True,
        gru_features: int = 512,
        use_timbre: bool = True,
        use_amp_latent: bool = True,
        log_pitch: bool = False,
        #synthesizer and reverb controls
        attenuate_gain: float = .02,
        harmonic_n_controls: int = 64,
        harmonic_norm_below_nyquist: bool = True,
        harmonic_amp_resample: str = "window",
        harmonic_angular_cumsum: bool = True,
        noise_filters: int = 65,
        noise_initial_bias: float = -5.0,
        noise_window_size: int = 257,
        reverb_trainable: bool = True,
        reverb_length_seconds: float = 1,
        reverb_add_dry: bool = True,
        reverb_ir: bool = None,
        mult_noise: bool = False,
        #timbre encoder parameters
        use_timbre_encoder: bool = True,
        use_reverb: bool = True,
        n_mfcc: int = 30,
        n_mels: int = 128,
        n_ftt: int = 1024,
        hop_length: int = 256,
        timbre_enc_size: int = 16
    ):
        super().__init__()
        self.sr = sr
        self.input_length_seconds = input_length_seconds
        self.synth_unit_frames = synth_unit_frames
        self.device = device
        self.mult_noise = mult_noise
        
        self.use_reverb = use_reverb
        self.reverb_length = int(reverb_length_seconds * sr)
        
        self.use_timbre_encoder = use_timbre_encoder
        
        if use_timbre_encoder:
            self.timbre_encoder = TimbreEncoder(sr = self.sr,
                                               n_mfcc=n_mfcc,
                                               n_mels = n_mels,
                                               n_ftt = n_ftt,
                                               hop_length = hop_length,
                                               gru_features = gru_features,
                                               gru_bidirectional = gru_bidirectional,
                                               timbre_enc_size = timbre_enc_size)
        
        self.decoder = GttDecoder(mlp_hidden_features = mlp_hidden_features,
                                    mlp_blocks = mlp_blocks,
                                    gru_bidirectional = gru_bidirectional,
                                    gru_features = gru_features,
                                    harmonic_n_controls = harmonic_n_controls,
                                    noise_filters = noise_filters,
                                    use_timbre_encoder = use_timbre_encoder,
                                    timbre_enc_size = timbre_enc_size,
                                    mult_noise = mult_noise,
                                    use_amp_latent = use_amp_latent,
                                    log_pitch = log_pitch)
        
        #intialize synthesizers
        self.harmonic_synth = AdditiveSynth(
                                    sample_rate=self.sr,
                                    normalize_below_nyquist=harmonic_norm_below_nyquist,
                                    amp_resample_method=harmonic_amp_resample,
                                    use_angular_cumsum=harmonic_angular_cumsum,
                                    frame_length=self.synth_unit_frames,
                                    attenuate_gain=attenuate_gain,
                                    device=self.device)
        
        self.noise_synth = FilteredNoise(
                                frame_length=self.synth_unit_frames,
                                attenuate_gain=attenuate_gain,
                                initial_bias=noise_initial_bias,
                                window_size=noise_window_size,
                                device=self.device,)
        
        self.reverb = Reverb(
                        trainable=reverb_trainable,
                        reverb_length=self.reverb_length,
                        add_dry=reverb_add_dry,
                        impulse_response=reverb_ir,
                        device=self.device,)
        
    
    def forward(self, x):
        """
        Input: dictionary of features
        output: (batch, output audio samples)
        """
        
        for key in x:
            x[key] = x[key].to(self.device)
        
        if self.use_timbre_encoder:
            x['timbre'] = self.timbre_encoder(x['audio'])
        
        x = self.decoder(x)
        harmonic_out = self.harmonic_synth(x)
        noise_out = self.noise_synth(x['noise'])
        
        if self.mult_noise:
            noise_out = noise_out + self.noise_synth(x['noise_2']) #test second noise
        
        if self.use_reverb:
            audio_out = self.reverb(harmonic_out + noise_out) #test second noise
        else:
            audio_out = harmonic_out + noise_out #test second noise
        
        #crop audio to match input length
        audio_out = audio_out[:,:self.sr*self.input_length_seconds]
        
        return audio_out
        