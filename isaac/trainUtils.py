import time
import math
import random
import os
from itertools import zip_longest

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

from torch import optim

from utils import asMinutes, timeSince, corpus_bleu
from langUtils import loadLangPairs, langDataset, langCollateFn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

SPECIAL_SYMBOLS_ID = PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3
NUM_SPECIAL = len(SPECIAL_SYMBOLS_ID)
BATCH_SIZE = 64
EARLY_STOP = 5
CLIP = 10

################################################################
##Fit
################################################################

def fit(train_loader, dev_loader, encoder, decoder, encoder_optim, decoder_optim, criterion, epochs, print_every, lang):
    start = time.time()
    print('Initializing')
    inp_lang, out_lang = loadLang(lang)
    losses, train_scores, dev_scores = [], [], []
    total_loss, early_stop = 0, EARLY_STOP
    for epoch in range(epochs):
        for i, (inp, inp_lens, output, out_mask, out_max) in enumerate(train_loader):
            loss = train(inp, inp_lens, output, out_mask, out_max, encoder, decoder, encoder_optim, decoder_optim, criterion)
            total_loss += loss
            if i % print_every == 0 & (i > 0):
                print_loss_avg = total_loss / print_every
                print("*************************************************")
                print("Epoch: {}".format(epoch))
                
                train_scores.append(bleuEval(encoder, decoder, train_loader, True, out_lang))
                dev_scores.append(bleuEval(encoder, decoder, dev_loader, False, out_lang))
                
                print("Time Elapsed: {} | Loss: {:.4}".format(asMinutes(time.time() - start), print_loss_avg))
                print("Train Score: {:.4} | Dev Score: {:.4}".format(train_scores[-1], dev_scores[-1]))
                print("*************************************************")
                print("")
                losses.append(print_loss_avg)
                total_loss = 0
        
                if len(losses) > 3:
                    if abs(losses[-1] - losses[-2]) < 0.00001:
                        early_stop -= 1
                    else:
                        early_stop = EARLY_STOP
        
        if early_stop == 0:
            print("Converged - Needed only {} Epochs".format(epoch))
            break
          
    return losses, train_scores, dev_scores

################################################################
##LoadLang
################################################################

def loadLang(lang):
    inp_lang, out_lang = loadLangPairs(lang)
    return inp_lang, out_lang

################################################################
##Train Epoch
################################################################

def train(inp, inp_lens, output, out_mask, out_max, encoder, decoder, encoder_optim, decoder_optim, criterion):
    
    loss, losses, n_totals = 0, [], 0
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    encoder_output, encoder_hidden = encoder(inp, inp_lens)
    
    decoder_input = torch.LongTensor([[SOS_ID for _ in range(inp.size(1))]]).to(device)
    decoder_hidden = encoder_hidden[decoder.n_layers:]
    
    for t in range(1, out_max):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
        decoder_input = output[t].view(1, -1)
        
        n_total = out_mask[t].sum()
        t_loss = criterion(decoder_output, output[t].to(device))
        loss += t_loss
        
        losses.append(t_loss.item() * n_total)
        n_totals += n_total

    loss.backward()
    
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), CLIP)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), CLIP)


    encoder_optim.step()
    decoder_optim.step()
    
    return (sum(losses)/n_totals).item() * 1.0

################################################################
##Eval
################################################################

def bleuEval(encoder, decoder, data_loader, print_translation, out_lang):
    with torch.no_grad():
        
        true_outputs = []
        decoder_outputs = []
        
        for i, (inp, inp_lens, output, out_mask, out_max) in enumerate(data_loader):
            if i * BATCH_SIZE >= 10000:
                break
            
            true_outputs.extend([[str(tok.item()) for tok in output[:,ind] if tok != PAD_ID] for ind in range(output.size(1))])
            encoder_output, encoder_hidden = encoder(inp, inp_lens)

            decoder_input = torch.LongTensor([[SOS_ID for _ in range(inp.size(1))]]).to(device)
            decoder_hidden = encoder_hidden[decoder.n_layers:]

            all_tokens = decoder_input.clone()
            #all_scores = torch.zeros([1,inp.size(1)], device=device)

            for idx in range(1, out_max):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
                decoder_scores, decoder_input = decoder_output.topk(1)

                all_tokens = torch.cat((all_tokens, decoder_input.t()), dim=0)
                #all_scores = torch.cat((all_scores, decoder_scores.t()), dim=0)

                decoder_input = decoder_input.t()

            for ind in range(all_tokens.size(1)):
                seq = []
                for tok in all_tokens[:,ind]:
                    if tok == EOS_ID:
                        seq.append(str(tok.item()))
                        break
                    elif tok == PAD_ID:
                        seq.append(str(EOS_ID))
                    else:
                        seq.append(str(tok.item()))


                decoder_outputs.extend([seq])
                
    if print_translation:
        print("True Translation: {}".format([out_lang.id2word[int(tok)] for tok in true_outputs[0]]))        
        print("Predicted Translation: {}".format([out_lang.id2word[int(tok)] for tok in decoder_outputs[0]]))
        print("")

    score = corpus_bleu(decoder_outputs, true_outputs, 4)
    return score