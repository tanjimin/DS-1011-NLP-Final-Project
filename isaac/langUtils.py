from itertools import zip_longest

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

from utils import directories, load_zipped_pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

SPECIAL_SYMBOLS_ID = PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3
NUM_SPECIAL = len(SPECIAL_SYMBOLS_ID)

data_dir, emb_dir = directories()

################################################################
##Language Class
################################################################

class Lang:
    def __init__(self, name, id2word, word2id, emb, learn_ids):
        self.name = name
        self.id2word = id2word
        self.word2id = word2id
        self.emb = emb

        self.n_words = len(id2word)
        self.learn_ids = learn_ids
        
    def addWords(self, train, dev, test):
        self.train = train
        self.dev = dev
        self.test = test
        
        self.max_length = int(np.percentile([len(s) for s in train], [85]).item())
        
    def addNums(self, train, dev, test):
        self.train_num = train
        self.dev_num = dev
        self.test_num = test
        


################################################################
##LanguagePairs
################################################################
        
def loadLangPairs(lang):
    if lang =="vi":
        inp = "vi"
    else:
        inp = "zh"
        
    #Input Language
    inpLang = Lang(inp, 
              load_zipped_pickle(emb_dir + "id2word_{}_dic.p".format(inp)), 
              load_zipped_pickle(emb_dir + "word2id_{}_dic.p".format(inp)), 
              load_zipped_pickle(emb_dir + "{}_embeddings_matrix_100K.p".format(inp)),
              SPECIAL_SYMBOLS_ID)

    inpLang.addWords(load_zipped_pickle(data_dir + "{}-en-tokens/train_{}_tok.p".format(inp, inp)),
               load_zipped_pickle(data_dir + "{}-en-tokens/dev_{}_tok.p".format(inp, inp)),
               load_zipped_pickle(data_dir + "{}-en-tokens/test_{}_tok.p".format(inp, inp)))

    inpLang.addNums(load_zipped_pickle(data_dir + "{}-en-tokens/train_{}_tok_num.p".format(inp, inp)),
               load_zipped_pickle(data_dir + "{}-en-tokens/dev_{}_tok_num.p".format(inp, inp)),
               load_zipped_pickle(data_dir + "{}-en-tokens/test_{}_tok_num.p".format(inp, inp)))

    #English Language
    enLang = Lang('en', 
              load_zipped_pickle(emb_dir + "id2word_en_dic.p"), 
              load_zipped_pickle(emb_dir + "word2id_en_dic.p"), 
              load_zipped_pickle(emb_dir + "en_embeddings_matrix_100K.p"),
              SPECIAL_SYMBOLS_ID)

    enLang.addWords(load_zipped_pickle(data_dir + "{}-en-tokens/train_en_tok.p".format(inp)),
               load_zipped_pickle(data_dir + "{}-en-tokens/dev_en_tok.p".format(inp)),
               load_zipped_pickle(data_dir + "{}-en-tokens/test_en_tok.p".format(inp)))

    enLang.addNums(load_zipped_pickle(data_dir + "{}-en-tokens/train_en_tok_num.p".format(inp)),
               load_zipped_pickle(data_dir + "{}-en-tokens/dev_en_tok_num.p".format(inp)),
               load_zipped_pickle(data_dir + "{}-en-tokens/test_en_tok_num.p".format(inp)))
    
    return inpLang, enLang


################################################################
##DataLoader
################################################################

###ADD CITATION

def zeroPadding(l, fillvalue=PAD_ID):
    return list(zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_ID):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_ID:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def inputVar(input_batch):
    lengths = np.array([len(i) for i in input_batch])
    padList = zeroPadding(input_batch)
    padVar = torch.LongTensor(padList, device=device)
    return padVar, lengths

def outputVar(output_batch):
    max_target_len = max([len(o) for o in output_batch])
    padList = zeroPadding(output_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask, device=device)
    padVar = torch.LongTensor(padList, device=device)
    return padVar, mask, max_target_len


class langDataset(Dataset):
    def __init__(self, data_tuple):
        self.inp, self.out = zip(*data_tuple)
        
        assert (len(self.inp) == len(self.out))

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, key):
        inp = self.inp[key]
        out = self.out[key]
        
        return [inp, out]

def langCollateFn(batch):
    inp_list, out_list = [], []
    
    for pair in batch:
        inp_list.append(pair[0])
        out_list.append(pair[1])
        
    inp, inp_lens =  inputVar(inp_list)
    order = inp_lens.argsort()[::-1].copy()
    inp = inp[:,order]
    inp_lens = inp_lens[order]
    
    output, out_mask, out_max = outputVar(out_list)
    output = output[:,order]
    out_mask = out_mask[:,order]
    
    return inp, torch.from_numpy(inp_lens).to(device), output, out_mask, out_max
    
################################################################
##Embeddings
################################################################    

###ADD CITATION

class HybridEmbeddings(nn.Module):
    def __init__(self, fixed_embeddings, learned_embeddings):
        super(HybridEmbeddings, self).__init__()
        self.fixed_embeddings = fixed_embeddings
        self.learned_embeddings = learned_embeddings

    @property
    def embedding_dim(self):
        return self.fixed_embeddings.embedding_dim

    def forward(self, ids_tensor):
        
        fixed_ids = ((ids_tensor - NUM_SPECIAL) * ((ids_tensor >= NUM_SPECIAL)).long()).to(device)
        learned_ids = ((ids_tensor) * ((ids_tensor < NUM_SPECIAL)).long()).to(device)
    
        embeddings = (
            self.fixed_embeddings(fixed_ids)
            + self.learned_embeddings(learned_ids)
        )
        return embeddings
    

def initHybridEmbeddings(raw_emb, learn_ids):

    raw_emb_fixed = raw_emb[[i for i in range(0, raw_emb.shape[0]) if i not in learn_ids],:]
    raw_emb_learn = raw_emb[[i for i in range(0, raw_emb.shape[0]) if i in learn_ids],:]
    
    fixed_embeddings = nn.Embedding(
            raw_emb_fixed.shape[0],
            raw_emb_fixed.shape[1],
            padding_idx=0,
        )
    fixed_embeddings.weight.data.copy_(
            torch.from_numpy(raw_emb_fixed))
    fixed_embeddings.weight.requires_grad = False
    learned_embeddings = nn.Embedding(
            len(learn_ids),
            raw_emb_fixed.shape[1],
            padding_idx=0,
        )
    learned_embeddings.weight.data.copy_(
            torch.from_numpy(raw_emb_learn))
    
    embeddings = HybridEmbeddings(
            fixed_embeddings=fixed_embeddings,
            learned_embeddings=learned_embeddings,
        )
    
    return embeddings

################################################################
##Embeddings
################################################################ 


def tensorToList(output):
    output_to_bleu = []
    
    for i in range(output.size(1)):
        output_to_bleu.append([str(j) for j in output[:,i].tolist()])
        
    return output_to_bleu

