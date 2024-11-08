from gtt.model import GttNet
from gtt.dataloader import FeatureExtractor
from gtt.utilities.utils import load_audio
import IPython.display as ipd 
import numpy as np
import torch
import torchaudio

#note predict function has not gone through thorough testing

def predict(input_audio_path, model, loudness_metrics, ckpt_path, midi_path='', save_path=''):
    """
    Predict audio from input audio file using trained model

    Args:
        input_audio_path: path to audio file
        model: GttNet to use for prediction
        ckpt_path : path to model weights (.pt)
        midi_path: optional path to midi file for pitch information
        save_path: optional path to save output audio file
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(torch.load(ckpt_path))
    
    model.eval()
    torch.cuda.empty_cache()
    
    sr = model.sr
    model_input_length_seconds = model.input_length_seconds
    
    audio = load_audio(input_audio_path, sr)
    
    #pad and segment input audio
    input_length = audio.shape[-1]
    segments = int(np.ceil(input_length/(sr*model_input_length_seconds)))
    pad_length = segments*sr*model_input_length_seconds - input_length
    padding = torch.zeros(1,int(pad_length))
    
    audio_padded = torch.cat((audio,padding),1)
    
    audio_padded = audio_padded.reshape(segments, -1)
    
    
    #get features 
    extr = FeatureExtractor(sr =  sr,
                                audio_dir = '',
                                trim_duration = 0,
                                max_n_pitch = 30,
                                loud_mean = loudness_metrics['mean'],
                                loud_std = loudness_metrics['std'],
                                device=device )
    
    audio_list = []
    amplitude_list = []
    pitch_list = []
    loudness_list = []
    
    for segment in range(segments):
        segment_features = extr.get_features(audio_padded[segment,:].unsqueeze(0), 
                                                         audio_start_index=segment*sr*model_input_length_seconds,
                                                         midi_path = midi_path)
        audio_list.append(segment_features['audio'].unsqueeze(0))
        pitch_list.append(segment_features['pitch'].unsqueeze(0))
        amplitude_list.append(segment_features['amplitude'].unsqueeze(0))
        loudness_list.append(segment_features['loudness'].unsqueeze(0))
    
    features = {'audio':torch.cat(audio_list,0),
               'pitch':torch.cat(pitch_list,0),
               'amplitude':torch.cat(amplitude_list,0),
               'loudness':torch.cat(loudness_list,0)} 
    
    with torch.no_grad():
        out_audio = model(features)
    
    out_audio = out_audio.reshape(1,-1)[:,:input_length]
    
    print('Input audio')
    ipd.display(ipd.Audio(audio.numpy(), rate=sr))
    
    print('Output audio')
    ipd.display(ipd.Audio(out_audio.cpu().numpy(), rate=sr))
    
    if save_path != '':
        print('saving file')
        torchaudio.save(save_path, out_audio.cpu(), sr)
    