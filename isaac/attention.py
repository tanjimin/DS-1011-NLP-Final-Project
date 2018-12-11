class Attn(torch.nn.Module):
    def __init__(self, params):
        super(Attn, self).__init__()
        self.method = params['method']
        self.hidden_size = params['hidden_size']
        
        #Define extra functions depending on method
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(self.hidden_size))
    
    def forward(self, hidden, encoder_outputs):
        if self.method == 'general':
            #h_t.T.dot(W_a.dot(h_s))
            attn_energies = torch.sum(hidden * self.attn(encoder_output), dim=2)
        elif self.method == 'concat':
            #v_a * tanh(w_a[h_t,h_s])
            energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
            attn_energies = torch.sum(self.v * energy, dim=2)
        elif self.method == 'dot':
            #h_t.T.dot(h_s)
            attn_energies = torch.sum(hidden * encoder_output, dim=2)

        return F.softmax(attn_energies.t(), dim=1).unsqueeze(1)

class AttnDecoderRNN(nn.Module):
    def __init__(self, params, raw_emb, learn_ids):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = params['attn_model']
        self.hidden_size = params['hidden_size']
        self.output_size = params['output_size']
        self.n_layers = params['n_layers']
        self.dropout = params['dropout']

        # Define layers
        self.embedding = initHybridEmbeddings(raw_emb, learn_ids)
        self.embed_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, dropout=(0 if n_layers == 1 else dropout), batch_first=True)
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.attn = Attn(self.attn_model, self.hidden_size)

    def forward(self, inp, hidden, encoder_output):
        #Embedding and dropout
        embedded = self.embedding(inp)
        embedded = self.embed_dropout(embedded)
        
        #GRU
        rnn_out, hidden = self.gru(embedded, hidden)
        
        #Attn weights * enc_output
        attn_weights = self.attn(rnn_output, encoder_output)
        context = attn_weights.bmm(encoder_output.transpose(0, 1))
        
        #Concat context to GRU output
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_inp = torch.cat((rnn_out, context), 1)
        concat_out = torch.tanh(self.concat(concat_inp))
        
        #Prediction
        output = self.out(concat_out)
        output = F.softmax(output, dim=1)
        
        return output, hidden