import time
import math
import random
import os
from itertools import zip_longest

import numpy as np
import pandas as pd

import pickle as pkl
import gzip

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

from torch import optim

from utils import asMinutes, timeSince, load_zipped_pickle, corpus_bleu, directories
from langUtils import loadLangPairs, langDataset, langCollateFn, initHybridEmbeddings, EncoderRNN, DecoderRNN
from trainUtils import train, fit, bleuEval

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns; sns.set()
sns.set_style("darkgrid")
sns.set_context("paper")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

data_dir, em_dir, fig_dir = directories()

SPECIAL_SYMBOLS_ID = PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3
NUM_SPECIAL = len(SPECIAL_SYMBOLS_ID)
BATCH_SIZE = 64

LEARNING_RATE = 0.01

vi, en = loadLangPairs("vi")

for j, i in enumerate(range(50, 350)):
    print("*******************************************************")
    print("*******************************************************")
    print("Running Hidden Size: {} | {:.3}% Complete".format(i, j/len(range(50, 350))))
    print("*******************************************************")
    print("*******************************************************")
    print("")
    #LOAD LANGS
    train_dataset = langDataset([(vi.train_num[i], en.train_num[i]) for i in range(len(vi.train_num)) if (len(vi.train[i]) < vi.max_length) & (len(en.train[i]) < en.max_length)])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               collate_fn=langCollateFn,
                                               shuffle=False)
    dev_dataset = langDataset([(vi.dev_num[i], en.dev_num[i]) for i in range(len(vi.dev_num)) if (len(vi.dev[i]) < vi.max_length) & (len(en.dev[i]) < en.max_length)])
    dev_loader = torch.utils.data.DataLoader(dataset=dev_dataset,
                                               batch_size=BATCH_SIZE,
                                               collate_fn=langCollateFn,
                                               shuffle=True)
    
    #SET PARAMS
    encoder_params = {'hidden_size':i, 'n_layers':1}
    decoder_params = {'hidden_size':encoder_params['hidden_size'], 'n_layers':1, 'output_size':en.n_words}

    encoder = EncoderRNN(encoder_params, vi.emb, vi.learn_ids).to(device)
    encoder_optim = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)

    decoder = DecoderRNN(decoder_params, en.emb, en.learn_ids).to(device)
    decoder_optim = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
    
    #SET CRITERION
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID).to(device)
    
    #FIT AND TRAIN
    losses, train_scores, dev_scores = fit(train_loader, dev_loader, encoder, decoder, encoder_optim, decoder_optim, criterion, 25, 300, "vi")
    
    #PLOT LOSSES
    plt.figure()

    pp = sns.lineplot(x = np.arange(len(losses)), y = losses, legend='brief')
    pp.set_title('Loss Over Time | Hidden Size: {}'.format(i))
    pp.set_ylabel("Loss")
    pp.set_xlabel("Time")
    pp.get_figure().savefig(fig_dir+"wo_att\\hs\\hs_{}_loss".format(i), bbox_inches='tight')
    
    #PLOT SCORES
    df = pd.concat([pd.DataFrame({'X':np.arange(len(train_scores)), 'Y':train_scores, 'Acc':'Train'}), 
                pd.DataFrame({'X':np.arange(len(dev_scores)), 'Y':dev_scores, 'Acc':'Dev'})], axis=0)


    plt.figure()
    pp = sns.lineplot(data=df, x = 'X', y = 'Y', hue='Acc', style="Acc", legend= "brief")
    pp.set_title("Scores")
    pp.set_ylabel("Accuracy")
    pp.set_xlabel("Epoch")
    pp.get_figure().savefig(fig_dir+"wo_att\\hs\\hs_{}_scores".format(i), bbox_inches='tight')
    
    torch.cuda.empty_cache()























