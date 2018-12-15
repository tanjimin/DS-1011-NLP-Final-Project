import pickle as pkl
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time

from langUtils import initHybridEmbeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, raw_emb, learn_ids):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        # input_size: input dictionary size
        self.embedding = initHybridEmbeddings(raw_emb, learn_ids)
#         self.embedding = nn.Embedding(input_size, hidden_size)
        self.num_layers = num_layers
        self.gru = nn.GRU(self.hidden_size, 
                          hidden_size, 
                          num_layers= num_layers, 
                          batch_first = True) # BATCH FIRST

    def forward(self, encoder_input, hidden_input):
        # encoder_input: batch * 1 (for 1 word each time)
        embedded_input = self.embedding(encoder_input)
        # embedded_input: batch * 1 * emb_dim
        # hidden_input: batch * 1(layer) * hidden_size
        output, hidden = self.gru(embedded_input, hidden_input)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=device)

class AttentionDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, max_length, batch_size, raw_emb, learn_ids, dropout_p=0.1):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        # Max length for a sentence
        self.max_length = max_length
        self.num_layers = num_layers
        
#         self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.embedding = initHybridEmbeddings(raw_emb, learn_ids)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, 
                          self.hidden_size,
                          num_layers= num_layers, 
                          batch_first = True) # BATCH_FRIST)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, decoder_input, hidden_input, encoder_hiddens):
        # hidden_input: 1 * batch * hidden_size
        hidden_input = hidden_input.squeeze(0)
        # decoder_input: batch * 1
        embedded_input = self.embedding(decoder_input)
        # embedded_input: batch * 1 * embed_size
        embedded_input = self.dropout(embedded_input).squeeze(1)
        
        # embedded_input: batch * embed_size
        # hidden_input: batch * hidden_size 
        # (Use input and newest hidden to decide which encoder hidden is important)
        attn_weights = F.softmax(self.attn(torch.cat((embedded_input, hidden_input), 1)), dim=1).unsqueeze(1)
        # encoder_output: max_length * batch * encoder_hidden_size
        encoder_hiddens_t = encoder_hiddens.transpose(0, 1)
        # attn_weights: batch * 1 * max_length(theoretical)
        cropped_attn_weights = attn_weights[:, :, :encoder_hiddens_t.shape[1]]
        # cropped_attn_weights: batch * 1 * max_length(actual)
        # encoder_hiddens_t: batch * max_length(actual) * encoder_hidden_size
        ## 
        attn_applied = torch.bmm(cropped_attn_weights, encoder_hiddens_t).squeeze(1)
        
        # embedded_input: batch * embed_size
        # attn_applied: batch * encoder_hidden_size
        output = torch.cat((embedded_input, attn_applied), 1)
        output = self.attn_combine(output)
        
        # output: batch * hidden_size
        gru_input = F.relu(output).unsqueeze(1)
        # hidden_input: batch * hidden_size
        hidden_input = hidden_input.unsqueeze(0)
        # gru_input: batch * 1 * hidden_size
        # hidden_input: 1 * batch * hidden_size
        output, hidden = self.gru(gru_input, hidden_input)
        output = self.out(output)
        #output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(self.num_layers,  self.batch_size, self.hidden_size, device=device)
