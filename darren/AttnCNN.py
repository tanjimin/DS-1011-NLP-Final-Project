import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from langUtils import initHybridEmbeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
