from itertools import zip_longest

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from utils import directories, load_zipped_pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

SPECIAL_SYMBOLS_ID = PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3
NUM_SPECIAL = len(SPECIAL_SYMBOLS_ID)

data_dir, emb_dir, fig_dir = directories()

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

##ADAPTED FROM https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

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
        
        if inp[-1] != 3:
            new_inp = inp[:-1].copy()
            new_inp.append(3)
        else:
            new_inp = inp.copy()
        
        if out[-1] != 3:
            new_out = out[:-1].copy()
            new_out.append(3)
        else:
            new_out = out.copy()
        
        return [new_inp, new_out]

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

###Modified version of https://github.com/zphang/usc_dae/blob/master/src/datasets/data.py#L185-L213

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
##Encoder
################################################################ 

##ADAPTED FROM https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html


class EncoderRNN(nn.Module):

    def __init__(self, params, raw_emb, learn_ids):
    
        super(EncoderRNN, self).__init__()
        
        self.hidden_size = params['hidden_size']
        self.n_layers = params['n_layers']
        
        self.embedding = initHybridEmbeddings(raw_emb, learn_ids)
        self.gru = nn.GRU(self.embedding.embedding_dim, params['hidden_size'], self.n_layers, bidirectional=True)
        
    def forward(self, inp, inp_lens):
        #Embed input
        embedded = self.embedding(inp)
        #Pack padded
        packed = pack_padded_sequence(embedded, inp_lens).to(device)
        
        #GRU
        output, hidden = self.gru(packed)
        #Pad packed
        output, _ = pad_packed_sequence(output)
        #Concat bidirectional layers
        output = output[:, :, :self.hidden_size] + output[:, : ,self.hidden_size:]
        return output, hidden

    
################################################################
##Basic Decoder
################################################################ 
    
    
class DecoderRNN(nn.Module):
    def __init__(self, params, raw_emb, learn_ids):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = params['hidden_size']
        self.n_layers = params['n_layers']
        self.output_size = params['output_size']
        
        self.embedding = initHybridEmbeddings(raw_emb, learn_ids)
        self.gru = nn.GRU(self.embedding.embedding_dim, params['hidden_size'], self.n_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inp, hidden, encoder_output=None):
        embedded = self.embedding(inp)
        embedded = F.relu(embedded)

        output, hidden = self.gru(embedded, hidden)
        output = self.out(output).squeeze(0)
        #output = F.softmax(output, dim=1).squeeze(0)
        return output, hidden

    
################################################################
##Luong Decoder
################################################################ 
    
##ADAPTED FROM https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    
class LuongAttnDecoder(nn.Module):
    def __init__(self, params, raw_emb, learn_ids):
        super(LuongAttnDecoder, self).__init__()
        
        self.hidden_size = params['hidden_size']
        self.output_size = params['output_size']
        self.n_layers = params['n_layers']
        self.dropout = params['dropout']

        # Define layers
        self.embedding = initHybridEmbeddings(raw_emb, learn_ids)
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(self.embedding.embedding_dim, self.hidden_size, self.n_layers, dropout=(0 if self.n_layers == 1 else self.dropout))
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inp, prev_hidden, encoder_output):
        embedded = self.embedding(inp)
        embedded = self.embedding_dropout(embedded)
        
        output, hidden = self.gru(embedded, prev_hidden)
        
        attn_energies = (torch.sum(output * encoder_output, dim=2)).t()
        attn_weights = F.softmax(attn_energies, dim=1).unsqueeze(1)
        
        context = attn_weights.bmm(encoder_output.transpose(0, 1)).squeeze(1)
        output = output.squeeze(0)
        
        concat_input = torch.cat((output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        output = self.out(concat_output)
        return output, hidden
