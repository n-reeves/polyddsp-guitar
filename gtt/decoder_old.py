import torch
from torch import nn
from torch.nn import functional as F
from gtt.utilities.utils import modified_sigmoid


#Note: dated decoder design produced in early stage of project. Closer resembels design in WIP repository for PolyDDSP and monophonic DDSP model
#This design is not used in final model and is included for reference only


class DecoderMLPBlock(nn.Module):
    """
    Args: 
        in_features: number of input features to block
        out_features: number of output features to block
        
    """
    def __init__(
        self,
        in_features: int,
        out_features: int = 512,
    ):
        super().__init__()
        self.dense = nn.Linear(in_features = in_features, out_features = out_features)
        self.layer_norm = nn.LayerNorm(out_features)
        self.act = nn.ReLU()
    
    def forward(self, x):
        """
        Input dim: (batch*voices, frames, in_features)
        Output dim: (batch*voices, frames, out_features)
        """
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.act(x)
        return x

    
class DecoderMLP(nn.Module):
    """
    Args:
        num_blocks: number of clocks in MLP
        in_features: numbers of input features
        hidden_features: number of hidden features (and output features)
        
    """
    def __init__(
        self,
        num_blocks: int = 3,
        in_features: int = 1,
        hidden_features: int = 512,
        
    ):
        super().__init__()
        
        self.mlp_blocks = nn.ModuleList()
        in_dim = in_features
        out_dim = hidden_features
        for i in range(num_blocks):
            self.mlp_blocks.append(DecoderMLPBlock(in_features = in_dim, out_features = out_dim))
            in_dim = out_dim
    
    def forward(self, x):
        """
        Input dim: (batch*voices, frames, in_features)
        Output dim: (batch*voices, frames, hidden_features)
        """
        
        for block in self.mlp_blocks:
            x = block(x)
        
        return x

    
class GttDecoder(nn.Module):
    """
    Args:
        
    
    Returns:
        
    """
    def __init__(
        self,
        mlp_hidden_features: int = 512,
        mlp_blocks: int = 3,
        gru_bidirectional: bool = True,
        gru_features: int = 512,
        harmonic_n_controls: int = 64,
        noise_filters: int = 65,
        use_timbre_encoder: bool = False,
        timbre_enc_size: int = 16,
        mult_noise: bool = False,
        log_pitch: bool = False,
        use_amp_latent: bool = True
    ):
        super().__init__()
        
        self.use_timbre_encoder = use_timbre_encoder
        self.use_amp_latent = use_amp_latent
        self.mult_noise = mult_noise
        self.log_pitch = log_pitch
        
        if use_timbre_encoder and use_amp_latent:
            self.gru_in_features = 4*mlp_hidden_features
        elif use_timbre_encoder or use_amp_latent:
            self.gru_in_features = 3*mlp_hidden_features
        else:
            self.gru_in_features = 2*mlp_hidden_features
        
        #MLPs applied to each input feature
        self.pitch_mlp = DecoderMLP(num_blocks = mlp_blocks, 
                                    hidden_features = mlp_hidden_features)
        self.loudness_mlp = DecoderMLP(num_blocks = mlp_blocks, 
                                       hidden_features = mlp_hidden_features)
        
        if self.use_amp_latent:
            self.amplitude_mlp = DecoderMLP(num_blocks = mlp_blocks, 
                                        hidden_features = mlp_hidden_features)
            
        
        if use_timbre_encoder:
            self.timbre_mlp = DecoderMLP(num_blocks = mlp_blocks,
                                         in_features = timbre_enc_size,
                                         hidden_features = mlp_hidden_features)
        
        #Gru applied to output of MLPs
        self.gru_out_features = 2*gru_features if gru_bidirectional else gru_features
        
        self.gru = nn.GRU(
            input_size=self.gru_in_features,
            hidden_size=gru_features,
            num_layers=1,
            batch_first=True,
            bidirectional=gru_bidirectional,
        )
        
        #MLP applied to concat of output from MLPs and output from gru
        self.out_mlp = DecoderMLP(num_blocks = mlp_blocks, 
                                       in_features = self.gru_in_features + self.gru_out_features,
                                       hidden_features = mlp_hidden_features)
        
        
        #output dense layers for amplitude residual, harmonic amplitudes, and noise filters
        self.harmonic_n_controls = harmonic_n_controls
        self.noise_filters = noise_filters
        
        self.dense_harmonic = nn.Linear(in_features = mlp_hidden_features,
                                       out_features = self.harmonic_n_controls+1)
        
        self.dense_noise = nn.Linear(in_features = mlp_hidden_features,
                                       out_features = self.noise_filters)
        
        if self.mult_noise:
            #test second noise
            self.dense_noise_2 = nn.Linear(in_features = mlp_hidden_features,
                                           out_features = self.noise_filters)
    
    def forward(self, x):
        """
        Input Format:
            pitch: (batch, voices,frames)
            amplitude: (batch, voices, frames)
            loudness: (batch, 1, frames)
            timbre (optional): (batch, voices, frames, num_timbre_features)
        
        """
        #get dimensions and reshape features for mlp
        [batch_size, voices, frames] = x['pitch'].shape
        
        pitch_in = x['pitch'].reshape(batch_size*voices, -1).unsqueeze(-1) #(batch*voices, frames, 1)
        
        if self.log_pitch:
            #apply log transformation to non zero values
            non_zero_mask = pitch_in!= 0
            log_pitch = torch.zeros_like(pitch_in)
            log_pitch[non_zero_mask] = torch.log(pitch_in[non_zero_mask])
            pitch_in = log_pitch
        
        loudness_exp = x['loudness'].expand(batch_size, voices, frames) # (batch, voices,frames)
        loudness_in = loudness_exp.reshape(batch_size*voices, -1).unsqueeze(-1) #(batch*voices, frames, 1)
        
        #apply MLP to input and concat
        pitch_latent = self.pitch_mlp(pitch_in)
        loudness_latent = self.loudness_mlp(loudness_in)
        
        latent = torch.cat((pitch_latent, loudness_latent), dim=2)
        
        if self.use_amp_latent:
            amplitude_in = x['amplitude'].reshape(batch_size*voices, -1).unsqueeze(-1) #(batch*voices, frames, 1)
            amplitude_latent = self.amplitude_mlp(amplitude_in)
            latent = torch.cat((latent, amplitude_latent), dim=2) #(batch*voices, frames, mlp_hidden_features*3 )
        
        if self.use_timbre_encoder:
            timbre_in = x['timbre'] #(batch, mfcc frames, timbre_enc_size)
            timbre_in = F.interpolate(timbre_in.permute(0,2,1), size=frames, mode='linear').permute(0,2,1) #(batch, frames, timbre enc size)
            
            [_, _, timbre_enc_size] = timbre_in.shape
            timbre_in = timbre_in.unsqueeze(1).expand(batch_size, voices, frames, timbre_enc_size) #(batch, voices, mfcc frames, timbre_enc_size)
            timbre_in = timbre_in.reshape(batch_size*voices, frames, timbre_enc_size) #(batc*voices, frames, timbre_enc_size)
            
            timbre_latent = self.timbre_mlp(timbre_in)
            latent = torch.cat((latent,timbre_latent), dim=2) #(batch*voices, frames, mlp_hidden_features*3 )
        
        gru_out = self.gru(latent)[0] #(batch*voices, frames, gru features (*2 if bidirectional)
        
        latent = torch.cat((latent, gru_out), dim=2) 
        
        pre_dense_out = self.out_mlp(latent) #(batch*voices, frames, mlp_hidden_features)
        
        harmonic_controls = self.dense_harmonic(pre_dense_out)  # (batch*voices, frames, harmonic controls + 1)
        
        harmonic_controls = harmonic_controls.reshape(batch_size, voices,frames, self.harmonic_n_controls+1) #(batch, voice, frames,harmonics + 1)
        
        
        amplitude_resid = harmonic_controls[...,:1].squeeze(-1) #(batches, voices, frames)
        harmonic_controls = harmonic_controls[...,1:].softmax(dim=-1).permute(0,1,3,2) #(batches, voices, harmonics, frames)
        
        x['harmonics'] = harmonic_controls
        modified_sigmoid
        #x['amplitude'] = self.amp_sig(amplitude_resid) #x['amplitude']
        x['amplitude'] = modified_sigmoid(amplitude_resid) #x['amplitude']
        
        
        noise_controls = self.dense_noise(pre_dense_out) # (batch*voices, frames, noise filters)
        #noise_controls = modified_sigmoid(noise_controls)
        
        # x['noise'] = noise_controls.softmax(dim=-1).reshape(batch_size, voices,frames, self.noise_filters).permute(0,1,3,2)
        x['noise'] = noise_controls.reshape(batch_size, voices,frames, self.noise_filters).permute(0,1,3,2) #(batch, voices, magnitude coefs, frames)
        
        if self.mult_noise:
            #test second noise
            noise_controls_2 = self.dense_noise_2(pre_dense_out) # (batch*voices, frames, noise filters)
            noise_controls_2 = modified_sigmoid(noise_controls_2)
            x['noise_2'] = (noise_controls + noise_controls_2).softmax(dim=-1).reshape(batch_size, voices,frames, self.noise_filters).permute(0,1,3,2) #(batches, voices, filters, frames)
        return x