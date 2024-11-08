import torch
from torch import nn
from torch.nn import functional as F
from gtt.utilities.utils import modified_sigmoid

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
        max_n_pitch: int = 30,
        gru_cat: bool = True,
        pre_dense_mlp: bool = True,
        use_amp_latent: bool = True,
        trainable_velocity: bool = True
    ):
        super().__init__()
        
        self.use_timbre_encoder = use_timbre_encoder
        self.use_amp_latent = use_amp_latent
        self.gru_cat = gru_cat
        self.pre_dense_mlp = pre_dense_mlp
        self.trainable_velocity = trainable_velocity
        
        if use_timbre_encoder and use_amp_latent:
            self.gru_in_features = 4*mlp_hidden_features
        elif use_timbre_encoder or use_amp_latent:
            self.gru_in_features = 3*mlp_hidden_features
        else:
            self.gru_in_features = 2*mlp_hidden_features
        
        #MLPs applied to each input feature
        self.f0_mlp = DecoderMLP(num_blocks = mlp_blocks, 
                                    hidden_features = mlp_hidden_features)
        
        self.f0_mlp_noise = DecoderMLP(num_blocks = mlp_blocks, 
                                    in_features = max_n_pitch,
                                    hidden_features = mlp_hidden_features)
        
        self.loudness_mlp = DecoderMLP(num_blocks = mlp_blocks, 
                                       hidden_features = mlp_hidden_features)
        
        self.loudness_mlp_noise = DecoderMLP(num_blocks = mlp_blocks, 
                                       hidden_features = mlp_hidden_features)
        
        if self.use_amp_latent:
            self.velocity_mlp = DecoderMLP(num_blocks = mlp_blocks, 
                                        hidden_features = mlp_hidden_features)
            
            self.velocity_mlp_noise = DecoderMLP(num_blocks = mlp_blocks, 
                                                 in_features = max_n_pitch,
                                                 hidden_features = mlp_hidden_features)
            
        
        if self.use_timbre_encoder:
            self.timbre_mlp = DecoderMLP(num_blocks = mlp_blocks,
                                         in_features = timbre_enc_size,
                                         hidden_features = mlp_hidden_features)
            
            self.timbre_mlp_noise = DecoderMLP(num_blocks = mlp_blocks,
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
        
        self.gru_noise = nn.GRU(
            input_size=self.gru_in_features,
            hidden_size=gru_features,
            num_layers=1,
            batch_first=True,
            bidirectional=gru_bidirectional,
        )
        
        #MLP applied to concat of output from MLPs and output from gru
        out_mlp_in_features = self.gru_in_features + self.gru_out_features if self.gru_cat else self.gru_out_features
        if self.pre_dense_mlp:
            self.out_mlp = DecoderMLP(num_blocks = mlp_blocks, 
                                           in_features = out_mlp_in_features,
                                           hidden_features = mlp_hidden_features)

            self.out_mlp_noise = DecoderMLP(num_blocks = mlp_blocks, 
                                           in_features = out_mlp_in_features,
                                           hidden_features = mlp_hidden_features)
        
        #output dense layers for amplitude residual, harmonic amplitudes, and noise filters
        self.harmonic_n_controls = harmonic_n_controls
        self.noise_filters = noise_filters
        
        dense_in_features = mlp_hidden_features if self.pre_dense_mlp else out_mlp_in_features
        
        self.dense_harmonic = nn.Linear(in_features = dense_in_features,
                                       out_features = self.harmonic_n_controls+1)


        dense_noise_out = self.noise_filters if self.trainable_velocity else self.noise_filters + 1
        
        self.dense_noise = nn.Linear(in_features = dense_in_features,
                                   out_features = dense_noise_out)
        
    
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
        
        f0_in_pitch = x['pitch'].reshape(batch_size*voices, -1).unsqueeze(-1) #(batch*voices, frames, 1)
        f0_in_noise = x['pitch'].permute(0,2,1) # (batch,frames, voices)
        
        loudness_in_pitch = x['loudness'].expand(batch_size, voices, frames) # (batch, voices,frames)
        loudness_in_pitch = loudness_in_pitch.reshape(batch_size*voices, -1).unsqueeze(-1) #(batch*voices, frames, 1)
        loudness_in_noise = x['loudness'].permute(0,2,1) #(batch,frames,1)
        
        #apply MLP to input and concat to get pitch latent
        f0_latent_pitch = self.f0_mlp(f0_in_pitch)
        loudness_latent_pitch = self.loudness_mlp(loudness_in_pitch)
        latent_pitch = torch.cat((f0_latent_pitch, loudness_latent_pitch), dim=2)
        
        #get noise latents
        f0_latent_noise = self.f0_mlp_noise(f0_in_noise) #(batch,frames, hidden features)
        loudness_latent_noise = self.loudness_mlp_noise(loudness_in_noise) #(batch,frames,hidden features)
        latent_noise = torch.cat((f0_latent_noise, loudness_latent_noise), dim=2)
        
        if self.use_amp_latent:
            amplitude_in = x['amplitude'].reshape(batch_size*voices, -1).unsqueeze(-1) #(batch*voices, frames, 1)
            amplitude_latent = self.velocity_mlp(amplitude_in)
            latent_pitch = torch.cat((latent_pitch, amplitude_latent), dim=2) #(batch*voices, frames, mlp_hidden_features*3 )
            
            amplitude_in_noise = x['amplitude'].permute(0,2,1) #batch,frames, voices
            amplitude_latent_noise = self.velocity_mlp_noise(amplitude_in_noise)
            latent_noise = torch.cat((latent_noise, amplitude_latent_noise), dim=2)
            
        if self.use_timbre_encoder:
            timbre_in = F.interpolate(x['timbre'].permute(0,2,1), size=frames, mode='linear').permute(0,2,1) #(batch, frames,timbre enc size)
            [_, _, timbre_enc_size] = timbre_in.shape
            
            timbre_in_pitch = timbre_in.unsqueeze(1).expand(batch_size, voices, frames, timbre_enc_size) #(batch, voices, frames, enc_size)
            timbre_in_pitch = timbre_in_pitch.reshape(batch_size*voices, frames, timbre_enc_size) #(batch*voices, frames, timbre_enc_size)
            
            timbre_latent_pitch = self.timbre_mlp(timbre_in_pitch)
            latent_pitch = torch.cat((latent_pitch, timbre_latent_pitch), dim=2) #batch*voices, frames, mlp_hidden_features*3 or 4 )
            
            timbre_latent_noise = self.timbre_mlp_noise(timbre_in) #batch, frames, hidden features
            latent_noise = torch.cat((latent_noise, timbre_latent_noise), dim=2) #batch, frames, hidfeatures*3 or 4
                                      
        gru_out = self.gru(latent_pitch)[0] #(batch*voices, frames, gru features (*2 if bidirectional)
        gru_out_noise = self.gru_noise(latent_noise)[0]
        
        #we cat input to gru and output similar to monophonic DDSP
        if self.gru_cat:
            latent_pitch = torch.cat((latent_pitch, gru_out), dim=2)
            latent_noise = torch.cat((latent_noise, gru_out_noise), dim=2)   
        else:
            latent_pitch = gru_out
            latent_noise = gru_out_noise
        
        if self.pre_dense_mlp:
            latent_pitch = self.out_mlp(latent_pitch) #(batch*voices, frames, mlp_hidden_features)
            latent_noise = self.out_mlp_noise(latent_noise)                         
       
        #get harmonic controls                             
        harmonic_controls = self.dense_harmonic(latent_pitch)  # (batch*voices, frames, harmonic controls + 1)
        harmonic_controls = harmonic_controls.reshape(batch_size, voices,frames, self.harmonic_n_controls+1) #(batch,voice,frames,harmonics+1)
        
        #amplitude learned independently for each voice rather than globally as amp envelopes applied pointwise to sins in Harmonic synth
        amplitude_resid = harmonic_controls[...,:1].squeeze(-1) #(batches, voices, frames)
        
        #note, could introduce midified sigmoid non linearity similar to monophonic DDSP
        harmonic_controls = harmonic_controls[...,1:].softmax(dim=-1).permute(0,1,3,2) #(batches, voices, harmonics, frames)
        
        x['harmonics'] = harmonic_controls
        
        #network learns to produce velocity controls
        if self.trainable_velocity:
            x['amplitude'] = modified_sigmoid(amplitude_resid)
            
            noise_controls = self.dense_noise(latent_noise) # (batch, frames, noise filters)
        else:
            noise_controls = self.dense_noise(latent_noise) # (batch, frames, noise filters + 1)
            
            global_amp_scale = noise_controls[...,:1].permute(0,2,1) #(batch, 1, frames)
            
            #apply global amp scaling to amplitudes prior to sigmoid 
            x['amplitude'] = modified_sigmoid(x['amplitude']*global_amp_scale)
            
            noise_controls = noise_controls[...,1:]
        
        #(batch, voices, magnitude coefs, frames)
        #exp sigmoid applied in module
        x['noise'] = noise_controls
        return x