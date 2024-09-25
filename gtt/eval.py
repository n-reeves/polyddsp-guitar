from frechet_audio_distance import FrechetAudioDistance 
from PolyDDSP.modules.losses import SpectralLoss
from gtt.utilities.utils import load_audio, crop_audio

import torch
import torchaudio
import os
import shutil
import IPython.display as ipd 
from PolyDDSP.modules.loudness import LoudnessExtractor


def test_model(model, test_loader, calc_fad=True, baseline_dir='',test_dir=''):
    """
    Runs Spectral loss and FAD evaluation (spectral loss score not used in evalutaion metheod)

    Args:
        model: GttNet
        test_loader: dataloader 
        calc_fad (bool, optional): whether or not calculate the FAD 
        baseline_dir (str, optional): folder containg audio files that act as target distribution for FAD calc
        test_dir: directory where resynthesized audio files are saved for FAD calc
    """
    msstft_criterion = SpectralLoss()
    resample_trans = torchaudio.transforms.Resample(orig_freq=model.sr, new_freq=16000)
    
    model.eval()
    msstft = 0
    save_counter = 0
    for i, batch in enumerate(test_loader):
        with torch.no_grad():
            out = model(batch)
            batch_msstft = msstft_criterion(out, batch['audio'])
        
        msstft += batch_msstft.item()
        
        if calc_fad:
            out_audio_batch = out.cpu()
            batch_count = out_audio_batch.shape[0]
            
            for segment_num in range(batch_count):
                segment_name ='{}.wav'.format(save_counter)
                file_save_dir = os.path.join(test_dir,segment_name) 
                save_counter += 1
                audio_segment = out_audio_batch[segment_num,:]
                
                audio_segment = resample_trans(audio_segment).unsqueeze(0)
                torchaudio.save(file_save_dir, audio_segment, 16000)
    
    example_in = batch['audio'][0,...]
    example_out = out[0,...]

    print('Example Input')
    ipd.display(ipd.Audio(example_in.cpu().numpy(), rate=model.sr))

    print('Example Output')
    ipd.display(ipd.Audio(example_out.detach().cpu().numpy(), rate=model.sr))
    
    print('Test Mean MSSTFT: {}'.format(msstft/len(test_loader)))
    
    if calc_fad:
        for file in os.listdir(baseline_dir):
            if file == '.ipynb_checkpoints':
                path = os.path.join(baseline_dir,file)
                
                shutil.rmtree(path)
                print("Removed: {}".format(path))
        frechet = FrechetAudioDistance(
                        #ckpt_dir="../checkpoints/vggish",
                        model_name="vggish",
                        sample_rate=16000,
                        use_pca=False,
                        use_activation=False,
                        verbose=False,
                        audio_load_worker=8)

        fad_score = frechet.score(baseline_dir, test_dir)
        print("FAD score: {}".format(fad_score))
        shutil.rmtree(test_dir)
        os.makedirs(test_dir)
    

def create_eval_baseline_dir(audio_dir, file_names, save_dir, segment_length=4, sr=16000):
    """
    Creates a directory of audio files that are cropped to a specified length to be used as baseline directory for FAD calculation
    
    Args:
        audio_dir str: source directory for audio files
        file_names: list of audio file names to be used in constructing baseline
        save_dir: directory in which to save cropped audio files
        segment_length (int, optional): cropped segment lengths in seconds
        sr (int, optional): the sample rate of the audio. Defaults to 16000 in agreement with requirements of frechet_audio_distance package 
            (FAD package resmaples automatically if necessary).
    """
    
    shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    audio_paths = []
    for file in file_names:
        audio_path = os.path.join(audio_dir,file)
        audio_paths.append(audio_path)
    
    for path in audio_paths:
        audio_name = os.path.basename(path)
        audio = load_audio(filepath=path, sr=sr)
        
        cropped_audio_dict = crop_audio(audio, segment_length=segment_length*sr)
        cropped_audio = cropped_audio_dict['audio']
        
        save_path = os.path.join(save_dir,audio_name)
        torchaudio.save(save_path, cropped_audio, sr)
        

def mfcc_fro_distance(input_audio,
                     output_audio,
                     sample_rate=22050,
                     n_mfcc=30,
                     n_fft=1024,
                     n_mels=128,
                    hop_length=256,
                    device='cpu'):
    """
    Calculate the Frobenius norm of the difference between the MFCCs of two tensors

    Args:
        input_audio: input audio to network
        output_audio: output audio from network
        sample_rate (int, optional): sample rate of audio files
        n_mfcc (int, optional): number of MFCCs created by transform
        n_fft (int, optional): number of fft bins
        n_mels (int, optional): number of mels used in MFCC transform
        hop_length (int, optional): hop length used in MFCC transform
        device (str, optional): device transform is placed on. shoudl be shared with input and output audio device

    Returns:
        Frobenius norm of the difference between the MFCCs of the input and output audio
    """
    mfcc_transform = torchaudio.transforms.MFCC(
                                            sample_rate=sample_rate,
                                            n_mfcc=n_mfcc,
                                            log_mels=False,
                                            melkwargs=dict(
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                n_mels=n_mels,
                                                f_min=20.0,
                                                f_max=8000.0,)
                                        ).to(device)
    in_mfcc = mfcc_transform(input_audio)
    out_mfcc = mfcc_transform(output_audio)
    
    
    norm = torch.linalg.matrix_norm(in_mfcc - out_mfcc, ord = 'fro').sum()
    
    return norm.item()
    
    
def loudness_l2_distance(input_audio,
                         output_audio,
                         sample_rate=22050,
                         frame_length=64,
                         device = 'cput'
                        ):
    """
    Calculate the L2 norm of the difference between the loudness of two audio files
    
    Args:
        input_audio: input audio to network
        output_audio: output audio from network
        sample_rate (int, optional): sample rate of audio files
        frame_length (int, optional): frame length used in loudness extraction
        device (str, optional): device transform is placed on. should be shared with input and output audio device
    """
    
    loudness_extractor = LoudnessExtractor(sr=sample_rate,
                                            frame_length=frame_length,
                                            device = device)
    
    input_loud = loudness_extractor(input_audio)
    output_loud = loudness_extractor(output_audio)
    
    norm = torch.linalg.vector_norm(input_loud -output_loud, ord=2)
    
    return norm.item()