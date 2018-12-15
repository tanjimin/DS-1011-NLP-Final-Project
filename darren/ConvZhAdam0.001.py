import pickle as pkl
import numpy as np
import gzip
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time

from utils import asMinutes, timeSince, load_zipped_pickle, corpus_bleu, directories
from langUtils import loadLangPairs, langDataset, langCollateFn, initHybridEmbeddings, tensorToList

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


vi, en = loadLangPairs("zh")
BATCH_SIZE = 32
train_dataset = langDataset([(vi.train_num[i], en.train_num[i]) for i in range(len(vi.train_num)) if (len(vi.train[i]) < vi.max_length) & (len(en.train[i]) < en.max_length)])
overfit_dataset = langDataset([(vi.train_num[i], en.train_num[i]) for i in range(32)])
overfit_loader = torch.utils.data.DataLoader(dataset=overfit_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=langCollateFn,
                                           shuffle=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=langCollateFn,
                                           shuffle=True)
dev_dataset = langDataset([(vi.dev_num[i], en.dev_num[i]) for i in range(len(vi.dev_num)) if (len(vi.dev[i]) < vi.max_length) & (len(en.dev[i]) < en.max_length)])
dev_loader = torch.utils.data.DataLoader(dataset=dev_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=langCollateFn,
                                           shuffle=True)
                                           
SPECIAL_SYMBOLS_ID = PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3


class ConvEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, max_len, batch_size, raw_emb, learn_ids, dropout=0.2,
                 num_channels_attn=512, num_channels_conv=512,
                 kernel_size=3, num_layers=5):
        super(ConvEncoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = embedding_size
        self.position_embedding = nn.Embedding(max_len, embedding_size)
        self.word_embedding = initHybridEmbeddings(raw_emb, learn_ids)
        self.num_layers = num_layers
        self.dropout = dropout

        self.conv = nn.ModuleList([nn.Conv1d(num_channels_conv, num_channels_conv, kernel_size,
                                      padding=kernel_size // 2) for _ in range(num_layers)])

    def forward(self, position_ids, sentence_as_wordids):
        # Retrieving position and word embeddings
        position_embedding = self.position_embedding(position_ids).unsqueeze(0)
        position_embedding = position_embedding.repeat(self.batch_size, 1, 1)
        word_embedding = self.word_embedding(sentence_as_wordids)
        
        # Applying dropout to the sum of position + word embeddings
        embedded = F.dropout(position_embedding + word_embedding, self.dropout, self.training)
        
        # Num Batches * Length * Channel ==> Num Batches * Channel * Length
        embedded = embedded.transpose(1, 2)
        
        # emdedded: Num Batches * Channel * Length
        cnn = embedded
        for i, layer in enumerate(self.conv):
            # layer(cnn) is the convolution operation on the input cnn after which
            # we add the original input creating a residual connection
            cnn = F.tanh(layer(cnn)+cnn)    
        return cnn        
        
class AttentionDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, max_length, batch_size, raw_emb, learn_ids, dropout_p=0.1):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.batch_size = batch_size
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
        
        
def trainAttention(inp, output, out_max, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, batch_size):
    total_avg_loss = 0
    loss = 0
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_len = inp.shape[1]
    encoder_outputs = torch.zeros(input_len, batch_size, 1, HIDDEN_SIZE, device=device)
    position_ids = torch.LongTensor(range(0, input_len)).to(device)
    encoder_conv_out = encoder(position_ids, inp)
    encoder_hiddens = encoder_conv_out.unsqueeze(1).transpose(0, 3).transpose(2,3)

    # Decode
    decoder_input = torch.tensor([SOS_ID] * batch_size, device=device)
    decoder_hidden = decoder.initHidden()

    # Always use Teacher Forcing
    for dc_idx in range(out_max):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input.unsqueeze(1), decoder_hidden, encoder_hiddens.squeeze(1))
        decoder_output = decoder_output.squeeze(1).to(device) # get rid of the seq dimention
        loss += criterion(decoder_output, output[:, dc_idx])
        decoder_input = output[:, dc_idx]

        ## Print Value
#         sample_sentence.append(torch.argmax(decoder_output[0]).item())

    loss.backward()
    total_avg_loss += loss.item() / out_max

    encoder_optimizer.step()
    decoder_optimizer.step()

    ## Print Value
#     print("Predict: ", ids2sentence(sample_sentence, en.id2word))
#     print("Actual: ", ids2sentence(output[0].cpu().numpy(), en.id2word))
        
    return total_avg_loss
#     return 0

def bleuEvalAttention(encoder, decoder, data_loader, batch_size):
    with torch.no_grad():
        true_outputs = []
        decoder_outputs = []
        for i, (inp, inp_lens, output, out_mask, out_max) in enumerate(data_loader):
            if i * batch_size >= 10000 or len(inp[0]) != batch_size:
                continue
            inp = inp.transpose(0,1).to(device)
            output = output.transpose(0,1).to(device)
            true_outputs.append([[str(tok.item()) for tok in out if tok != 0] for out in output])
            input_len = inp.shape[1]
            encoder_outputs = torch.zeros(input_len, batch_size, 1, HIDDEN_SIZE, device=device)
            position_ids = torch.LongTensor(range(0, input_len)).to(device)
            encoder_conv_out = encoder(position_ids, inp)
            encoder_hiddens = encoder_conv_out.unsqueeze(1).transpose(0,3).transpose(2,3)

            # Decode
            decoder_input = torch.tensor([SOS_ID] * batch_size, device=device)
            decoder_hidden = decoder.initHidden()

            # Greedy
            for dc_idx in range(out_max):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input.unsqueeze(1), decoder_hidden, encoder_hiddens.squeeze(1))
                decoder_output = decoder_output.squeeze(1).to(device) # get rid of the seq dimention
                topv, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([topi[i][0] for i in range(inp.size(0))]).to(device)
                ## Print Value
                decoder_outputs.append(list(decoder_input.cpu().numpy()))
            ## Print Value
        predict = []
        for seq in np.array(decoder_outputs).T.astype(str):
            seq_toks = []
            for tok in seq:
                seq_toks.append(tok)
                if tok == '3':
                    break
            predict.append(seq_toks)
#         print('Sample True: ', ' '.join([en.id2word[int(i)] for i in true_outputs[0][0]]))
#         print('Sample Predicted: ', ' '.join([en.id2word[int(i)] for i in predict[0]]))
#         for seq in predict:
#             print('Sample Predicted: ', ' '.join([en.id2word[int(i)] for i in seq]))
        bleu_score = corpus_bleu(predict, true_outputs, 4)
        return bleu_score
        
        
def fitAttention(train_loader, dev_loader, encoder, decoder, encoder_opt, decoder_opt, criterion, batch_size, epochs, print_every):
    start = time.time()
    print('Initializing Model Training + Eval...')
    losses = []
    train_scores = []
    dev_scores = []
    for epoch in range(epochs):
        loss = 0
        for i, (inp, inp_lens, output, out_mask, out_max) in enumerate(train_loader):
            if (len(inp[0]) != batch_size):
                continue
            inp.transpose_(0,1)
            output.transpose_(0,1)
            inp = inp.to(device)
            output = output.to(device)
            loss += trainAttention(inp, output, out_max, encoder, decoder, encoder_opt, decoder_opt, criterion, batch_size)
            if i % print_every == 0 and i > 0:
                pkl.dump(encoder, open('zh-conv-encoder-adam0.001.p', 'wb'))
                pkl.dump(decoder, open('zh-conv-decoder-adam0.001.p', 'wb'))
                losses.append(loss/i)
                print("Time Elapsed: {} | Loss: {:.4}".format(asMinutes(time.time() - start),
                                                                                loss/i))
        train_score = bleuEvalAttention(encoder, decoder, train_loader, batch_size)
#         dev_score = bleuEvalATtention(encoder, decoder, dev_loader, batch_size)
        train_scores.append(train_score)
#         dev_scores.append(dev_score)
        print("Epoch: {} | Time Elapsed: {} | Loss: {:.4} | Train BLEU: {:.4} | Dev BLEU: ".format(epoch + 1, 
                                                                                                        asMinutes(time.time() - start),
                                                                                                        loss/len(train_loader), 
                                                                                                        train_score))
#                                                                                                         dev_score))

# dic_size_vi = len(id2word_vi_dic.keys())
# dic_size_en = len(id2word_en_dic.keys())
HIDDEN_SIZE = 300
LEARNING_RATE = 0.001
MAX_LENGTH = 100
## Add ignore index
criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

encoder = ConvEncoder(vocab_size = vi.n_words, embedding_size = 300, max_len=MAX_LENGTH, batch_size = BATCH_SIZE, dropout=0.2, num_channels_attn=HIDDEN_SIZE, num_channels_conv=300, raw_emb = vi.emb, learn_ids = vi.learn_ids).to(device)

decoder = AttentionDecoderRNN(hidden_size = HIDDEN_SIZE, output_size = en.n_words, num_layers = 1, max_length = MAX_LENGTH, batch_size = BATCH_SIZE, raw_emb = en.emb, learn_ids = en.learn_ids, dropout_p=0.1).to(device)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)


fitAttention(train_loader, dev_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, BATCH_SIZE, 15, 300)
