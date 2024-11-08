import torch
from torch.nn import functional as F
from torch import nn, optim

import os
import time
import IPython.display as ipd 
import matplotlib.pyplot as plt

from PolyDDSP.modules.losses import SpectralLoss
from gtt.eval import mfcc_fro_distance, loudness_l2_distance



def train_epoch(model, train_dataloader, optimizer, criterion, loud_batch=False, loud_epoch=True):
    """
    Runs a single epoch of training on the model

    Args:
        model: torch model
        train_dataloader: datalaoder used to train network
        optimizer: optimzer used to update model weights
        criterion: loss function
        loud_batch (bool, optional): whether ot not to display output from every 100 batches. used to debug. Defaults to False.
        loud_epoch: whether or not to display audio examples of inputs and outputs from last batch of epoch

    Returns:
        _type_: total loss for the epoch
    """
    model.train()
    epoch_loss = 0
    last_batch_loss = 1000000
    for i, batch in enumerate(train_dataloader):
        input_audio = batch['audio']
        
        optimizer.zero_grad()
        output_audio = model(batch)
            
        loss = criterion(output_audio,input_audio)
        
        #accounts for midi gaps in fully silent segments produced during random croppings
        #weights are not updated when input is fully silent
        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        #debug exploding gradient
        if loss.item() > last_batch_loss*5:
            print('batch loss increased by 5x')
            
            for i in range(input_audio.shape[0]):
                example_in = input_audio[i,...]
                example_out = output_audio[i,...]
                
                print('Example Input')
                ipd.display(ipd.Audio(example_in.detach().cpu().numpy(), rate=model.sr))

                print('Example Output')
                ipd.display(ipd.Audio(example_out.detach().cpu().numpy(), rate=model.sr))
        
        
        last_batch_loss = loss.item()
        if i % 100 == 0 and loud_batch:
            print('Batch {0} Loss: {1}'.format(i,loss.item()))
            
            example_in = input_audio[0,...]
            example_out = output_audio[0,...]
            
            print('Example Input')
            ipd.display(ipd.Audio(example_in.detach().cpu().numpy(), rate=model.sr))
            
            print('Example Output')
            ipd.display(ipd.Audio(example_out.detach().cpu().numpy(), rate=model.sr))
            
            plt.imshow(batch['amplitude'][0,...].detach().cpu().numpy())
            plt.colorbar()
            plt.show()
    
    if loud_epoch:
        example_in = input_audio[0,...]
        example_out = output_audio[0,...]
        print('Example Input')
        ipd.display(ipd.Audio(example_in.detach().cpu().numpy(), rate=model.sr))

        print('Example Output')
        ipd.display(ipd.Audio(example_out.detach().cpu().numpy(), rate=model.sr))
        
    return epoch_loss


def train_loop(model, 
                train_loader, 
                valid_loader, 
                epochs=100,
                valid_freq=5,
                ckpt_dir='',
                loud_batch=False,
                loud_epoch_freq=1,
                train_hours=8,
                early_stop_epochs=20
              ):
    """
    General training loop function

    Args:
        model : network to train
        train_loader : training data loader
        valid_loader : validation data loader
        epochs : number of epochs to train for
        valid_freq: how frequently to run the network on the validation set
        ckpt_dir: directory where checkpoints are saved
        loud_batch: whether or not audio outputs fromevery 100 batches should be displayed during training epochs
        loud_epoch_freq (int, optional): epoch frequency of audio examples to be displayed
        train_hours (int, optional): Number of hours to train the network for
        early_stop_epochs (int, optional): stop the network after this many epochs of no decrease in loss
    """
    
    
    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=.0001)
    criterion = SpectralLoss()
    
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            factor=0.85,
                                                            patience=5)
    
    #placeholder, values assigned after 0 epoch
    mfcc_dist_min = 0
    no_mfcc_dist_mint = True
    stagnate_counter = 0
    min_epoch_loss = 0
    
    start_time = time.time()
    for epoch in range(epochs):
            
        print('Starting epoch {}'.format(epoch+1))
        
        if epoch%loud_epoch_freq == 0:
            loud_epoch = True
        else:
            loud_epoch = False
        
        epoch_loss = train_epoch(model, 
                                 train_loader, 
                                 optimizer, 
                                 criterion, 
                                 loud_batch=loud_batch,
                                 loud_epoch=loud_epoch)
        
        scheduler.step(epoch_loss)
        print('Epoch Loss: {}'.format(epoch_loss))
        
        if epoch_loss >= min_epoch_loss and epoch > 0:
            stagnate_counter += 1
        
        if min_epoch_loss >= epoch_loss or epoch == 0:
            min_epoch_loss = epoch_loss
            stagnate_counter = 0
        
        if (epoch+1)%valid_freq == 0:
            print('Starting Validation')
            
            model.eval()
            msstft = 0
            mfcc_dist = 0
            for i, batch in enumerate(valid_loader):
                with torch.no_grad():
                    eval_out = model(batch)
                
                batch_msstft = criterion(eval_out, batch['audio'])
                msstft += batch_msstft.item()
                
                batch_mfcc_dist = mfcc_fro_distance(batch['audio'], eval_out, device=model.device)
                mfcc_dist += batch_mfcc_dist

            print('Validation Multiscale Spectral Loss: {}'.format(msstft))
            print('Validation MFCC Frobenius Norm of Distance: {}'.format(mfcc_dist))
            
            example_in = batch['audio'][0,...]
            example_out = eval_out[0,...]
            
            print('Example Input')
            ipd.display(ipd.Audio(example_in.detach().cpu().numpy(), rate=model.sr))
            
            print('Example Output')
            ipd.display(ipd.Audio(example_out.detach().cpu().numpy(), rate=model.sr))
            
            if no_mfcc_dist_mint or mfcc_dist < mfcc_dist_min:
                print('Saving Model')
                mfcc_dist_min = mfcc_dist
                no_mfcc_dist_mint = False
                
                save_path = os.path.join(ckpt_dir,'model_epoch.pt')
                torch.save(model.state_dict(), save_path)
            
        end_epoch_time = time.time()
        total_time_seconds = end_epoch_time - start_time
        total_time_hrs = total_time_seconds/3600
        
        
        if total_time_hrs > train_hours or stagnate_counter >= early_stop_epochs:
            break
            
                