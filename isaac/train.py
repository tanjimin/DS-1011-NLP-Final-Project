import os
import time
import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from utils import *

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns; sns.set()
sns.set_style("darkgrid")
sns.set_context("paper")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPECIAL_SYMBOLS_ID = \
    PAD_ID, UNK_ID, SOS_ID, EOS_ID = \
    0, 1, 2, 3
NUM_SPECIAL = len(SPECIAL_SYMBOLS_ID)

##################################################################
##################################################################
##################################################################

class ModelTrain():
    def __init__(self, data, encoder, decoder, e_optim, d_optim, criterion, tfr, i_len, o_len):
        
        self.data = data
        
        self.encoder = encoder
        self.decoder = decoder
        
        self.e_optim = e_optim 
        self.d_optim = d_optim 
        
        self.criterion = criterion
        
        self.tfr = tfr
        self.i_len = i_len
        self.o_len = o_len
        
    def train(self, input_tensor, target_tensor):
        self.e_optim.zero_grad()
        self.d_optim.zero_grad()
        
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        
        e_outputs = torch.zeros(self.i_len, self.encoder.hidden_size, device=device)        
        e_hidden = encoder.initHidden()
        
        loss = 0
        
        for ei in range(i_length):
            e_output, e_hidden = self.encoder(input_tensor[ei], e_hidden)
            e_outputs[ei] = e_output[0, 0]
        
        d_hidden = e_hidden
        d_input = torch.tensor([[SOS_ID]], device=device)

        if random.random() < self.tfr:
            for di in range(target_length):
                d_output, d_hidden, d_attention = self.decoder(d_input, d_hidden, e_outputs)
                loss += self.criterion(d_output, target_tensor[di])
                d_input = target_tensor[di] 

        else:
            for di in range(target_length):
                d_output, d_hidden, d_attention = self.decoder(d_input, d_hidden, e_outputs)
                topv, topi = d_output.topk(1)
                d_input = topi.squeeze().detach()  
                loss += self.criterion(d_output, target_tensor[di])
                if d_input.item() == EOS_ID:
                    break

        loss.backward()

        self.e_optim.step()
        self.d_optim.step()

        return loss.item() / target_length
    
    def train(self, n_iters, batch_size, print_every=1000, plot_every=100):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
    
        training_pairs = [[random.choice(self.data) for j in batch_size] for i in range(n_iters)]

        for iter in range(1, n_iters + 1):
            
            loss_batch = 0
            
            for b in range(batch_size):
                training_pair = training_pairs[iter - 1][b]
                input_tensor = training_pair[0]
                target_tensor = training_pair[1]

                loss = self.train(input_tensor, target_tensor)
                loss_batch += loss
                
            print_loss_total += loss_batch / batch_size #DOUBLE CHECK THIS
            plot_loss_total += loss_batch / batch_size #DOUBLE CHECK THIS

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
        
        self.n_iters = n_iters
        self.losses = plot_losses
        
    def showPlot(self):
        plt.figure()
        fig = plt.figure(figsize=(10,6))
        fig_plt = sns.lineplot(x=np.arange(0, self.n_iters, int(self.n_iters/len(self.losses))), y=self.losses)
        fig_plt.set_title("Loss Over Time")
        fig_plt.set_ylabel("Loss")
        fig_plt.set_xlabel("Iterations")
        return fig_plt.get_figure()
        
    def evaluate(self, sentence, id2word):
        with torch.no_grad():
            input_tensor = sentence
            input_length = input_tensor.size()[0]

            encoder_outputs = torch.zeros(self.i_len, self.encoder.hidden_size, device=device)
            encoder_hidden = encoder.initHidden()
            
            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_ID]], device=device)  # SOS
            decoder_hidden = encoder_hidden
            decoded_words = []
            decoder_attentions = torch.zeros(self.o_len, self.o_len)

            for di in range(self.o_len):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                if type(decoder_attention) != str:
                    decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_ID:
                    decoded_words.append('<eos>')
                    break
                else:
                    decoded_words.append(id2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]