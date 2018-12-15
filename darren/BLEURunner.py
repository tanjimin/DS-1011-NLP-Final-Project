#from AttnCNN import ConvEncoder, AttentionDecoderRNN
from AttnRNN import EncoderRNN, AttentionDecoderRNN
import BLEUs
import pickle as pkl
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
from utils import asMinutes, timeSince, load_zipped_pickle, corpus_bleu, directories
from langUtils import loadLangPairs, langDataset, langCollateFn, initHybridEmbeddings, tensorToList
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vi, vi_en = loadLangPairs("vi")
zh, zh_en = loadLangPairs("zh")
BATCH_SIZE = 32

vi_test_dataset = langDataset([(vi.test_num[i], vi_en.test_num[i]) for i in range(len(vi.test_num)) if (2 < len(vi.test[i]) < vi.max_length) & (2 < len(vi_en.test[i]) < vi_en.max_length)])
vi_test_loader = torch.utils.data.DataLoader(dataset=vi_test_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=langCollateFn,
                                           shuffle=True)
# overfit_dataset = langDataset([(vi.train_num[i], vi_en.train_num[i]) for i in range(32)])
# overfit_loader = torch.utils.data.DataLoader(dataset=overfit_dataset,
#                                            batch_size=BATCH_SIZE,
#                                            collate_fn=langCollateFn,
#                                            shuffle=True)

zh_test_dataset = langDataset([(zh.test_num[i], zh_en.test_num[i]) for i in range(len(zh.test_num)) if (2 < len(zh.test[i]) < zh.max_length) & (2 < len(zh_en.test[i]) < zh_en.max_length)])
zh_test_loader = torch.utils.data.DataLoader(dataset=vi_test_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=langCollateFn,
                                           shuffle=True)


#zh_attn_sgd_encoder = pkl.load(open('zh-g-attn-encoder-sgd0.01.p', 'rb'))
#zh_attn_sgd_decoder = pkl.load(open('zh-g-attn-decoder-sgd0.01.p', 'rb'))
#print('--------ZH to EN RNN ATTN SGD 0.01----------')
#print(BLEUs.bleuEvalAttention(zh_attn_sgd_encoder, zh_attn_sgd_decoder, zh_test_loader, 1))
#print(BLEUs.bleuEvalAttention(zh_attn_sgd_encoder, zh_attn_sgd_decoder, zh_test_loader, 5))

print('--------ZH to EN RNN ATTN ADAM 0.01---------')
zh_attn_adam_encoder = pkl.load(open('zh-g-attn-encoder-adam0.01.p', 'rb'))
zh_attn_adam_decoder = pkl.load(open('zh-g-attn-decoder-adam0.01.p', 'rb'))
print('Beam 1: ', BLEUs.bleuScoreAttention(zh_test_loader, zh_attn_adam_encoder, zh_attn_adam_decoder, BATCH_SIZE, 300, 1))
print('Beam 5: ', BLEUs.bleuScoreAttention(zh_test_loader, zh_attn_adam_encoder, zh_attn_adam_decoder, BATCH_SIZE, 300, 5))


#zh_conv_adam_encoder = pkl.load(open('zh-conv-encoder-adam0.001.p', 'rb'))
#zh_conv_adam_decoder = pkl.load(open('zh-conv-decoder-adam0.001.p', 'rb'))
#print(BLEUs.bleuScoreConv(zh_test_loader, zh_conv_adam_encoder, zh_conv_adam_decoder, BATCH_SIZE, 300, 1))
#print(BLEUs.bleuScoreConv(zh_test_loader, zh_conv_adam_encoder, zh_conv_adam_decoder, BATCH_SIZE, 300, 5))

#vi_conv_adam_encoder = pkl.load(open('vi-conv-encoder-adam0.001.p', 'rb'))
#vi_conv_adam_decoder = pkl.load(open('vi-conv-decoder-adam0.001.p', 'rb'))
#print(BLEUs.bleuScoreConv(vi_test_loader, vi_conv_adam_encoder, vi_conv_adam_decoder, BATCH_SIZE, 300, 1))
#print(BLEUs.bleuScoreConv(vi_test_loader, vi_conv_adam_encoder, vi_conv_adam_decoder, BATCH_SIZE, 300, 5))

#zh_conv_sgd_encoder = pkl.load(open('zh-conv-encoder-sgd0.01.p', 'rb'))
#zh_conv_sgd_decoder = pkl.load(open('zh-conv-decoder-sgd0.01.p', 'rb'))
#print(BLEUs.bleuScoreConv(zh_conv_sgd_encoder, zh_conv_sgd_decoder, zh_test_loader, 1))
#print(BLEUs.bleuScoreConv(zh_conv_sgd_encoder, zh_conv_sgd_decoder, zh_test_loader, 5))

#vi_conv_sgd_encoder = pkl.load(open('vi-conv-encoder-sgd0.01.p', 'rb'))
#vi_conv_sgd_decoder = pkl.load(open('vi-conv-decoder-sgd0.01.p', 'rb'))
#print(BLEUs.bleuScoreConv(vi_conv_sgd_encoder, vi_conv_sgd_decoder, vi_test_loader, 1))
#print(BLEUs.bleuScoreConv(vi_conv_sgd_encoder, vi_conv_sgd_decoder, vi_test_loader, 5))
