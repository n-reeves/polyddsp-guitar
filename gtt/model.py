import torch
from torch import nn
import torchaudio

from gtt.decoder import GttDecoder
from gtt.timbre.timbre_encoder import TimbreEncoder

from PolyDDSP.modules.synth import AdditiveSynth
from PolyDDSP.modules.noise import FilteredNoise
from PolyDDSP.modules.reverb import Reverb

class GttNet(nn.Module):
    """
    Args:
        sr: sample rate of input audio
        input_length_seconds: length of input audio in seconds
        synth_unit_frames: frames synthesized per input feature
        device: model device
        dict_out: whether or not to return dictionary of outputs from harmonic and noise synthesizers
        mlp_hidden_features: hidden features in mlp layers
        mlp_blocks: number of mlp blocks
        gru_bidirectional: whether or not gru is bidirectional
        gru_features: number of features in gru
        use_amp_latent: whether or not to use MLP to produce amplitude latent
        max_n_pitch: maximum number of F0s in audio segment (corresponds to number of voices used for synthesis)
        gru_cat: whether or not to concat output from MLPs with output from GRU prior to dense layers
        pre_dense_mlp: whether or not to use MLP prior to dense layers
        trainable_velocity: whether or not to produce trainable velocity controls (True = configuration 2 in the accompanying paper)
        attenuate_gain: gain multiplier applied at end of generation
        harmonic_n_controls: number of harmonics for each voice
        harmonic_norm_below_nyquist: normalize harmonics below nyquist
        harmonic_amp_resample: method of resampling amplitude envelopes
        harmonic_angular_cumsum: whether to use angular cumulative sum
        noise_filters: number of filters used in noise synthesizer
        noise_initial_bias: initial shift applied to dense out for noise synthesizer prior to sigmoid activation
        noise_window_size: size of window used in FIR filter for noise synthesizer
        reverb_trainable: whether or not reverb is trainable
        reverb_length_seconds: reverb length in seconds
        reverb_add_dry: whether or not to add dry signal to reverb output
        reverb_ir: whether or not to use impulse response for reverb
        only_harmonic: whether or not to return only the output from the harmonic synthesizer
        use_timbre_encoder: enable timbre encoder
        use_reverb: apply reverb to sum of noise and harmonic signals
        n_mfcc: number of MFCC features used by timbre encoder
        n_mels: number of mels used in MFCC transform by timbre encoder
        n_ftt: number of FFT bins used in MFCC transform by timbre encoder
        hop_length: hop length used in MFCC transform by timbre encoder
        timbre_enc_size: size of timbre latent
    Return:
        (batch, samples)
    
    """
    def __init__(
        self,
        sr: int = 22050,
        input_length_seconds: int = 4,
        synth_unit_frames: int = 64,
        device: str = "cpu",
        dict_out: bool = False,
        #decoder parameters
        mlp_hidden_features: int = 512,
        mlp_blocks: int = 3,
        gru_bidirectional: bool = True,
        gru_features: int = 512,
        use_amp_latent: bool = True,
        max_n_pitch: int = 30,
        gru_cat: bool = False,
        pre_dense_mlp: bool = True,
        trainable_velocity: bool = True,
        #synthesizer and reverb controls
        attenuate_gain: float = .02,
        harmonic_n_controls: int = 101,
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
        only_harmonic: bool = False,
        #timbre encoder parameters
        use_timbre_encoder: bool = True,
        use_reverb: bool = False,
        n_mfcc: int = 30,
        n_mels: int = 128,
        n_ftt: int = 1024,
        hop_length: int = 128,
        timbre_enc_size: int = 15
    ):
        super().__init__()
        self.sr = sr
        self.input_length_seconds = input_length_seconds
        self.synth_unit_frames = synth_unit_frames
        self.device = device
        self.only_harmonic = only_harmonic
        self.trainable_velocity = trainable_velocity
        
        self.use_reverb = use_reverb
        self.reverb_length = int(reverb_length_seconds * sr)
        
        self.use_timbre_encoder = use_timbre_encoder
        self.dict_out = dict_out
        
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
                                    use_amp_latent = use_amp_latent,
                                    gru_cat = gru_cat,
                                    pre_dense_mlp = pre_dense_mlp,
                                    trainable_velocity = trainable_velocity,
                                    max_n_pitch = max_n_pitch)
        
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
        Input: dictionary of features produced by feature extractor
        output: torch tensor of shape (batch, output audio samples)
        """
        
        for key in x:
            x[key] = x[key].to(self.device)
        
        if self.use_timbre_encoder:
            x['timbre'] = self.timbre_encoder(x['audio'])
        
        x = self.decoder(x)
        harmonic_out = self.harmonic_synth(x)
        noise_out = self.noise_synth(x['noise'])
        
        if self.only_harmonic:
            out_signal = harmonic_out
        else:
            out_signal = harmonic_out + noise_out
    
        if self.use_reverb:
            out_signal = self.reverb(out_signal)
        
        #crop audio to match input length
        out = out_signal[:,:self.sr*self.input_length_seconds]
        
        if self.dict_out:
            out = {'harmonic':harmonic_out[:,:self.sr*self.input_length_seconds], 
                  'noise': noise_out[:,:self.sr*self.input_length_seconds],
                  'out':out_signal }
        
        return out
        