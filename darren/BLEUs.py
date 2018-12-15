import bisect
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
class SortedList(list):
    def insort(self, x):
        bisect.insort(self, x)
        
SPECIAL_SYMBOLS_ID = PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3
def bleuScore(data_loader, encoder, decoder, batch_size, HIDDEN_SIZE, beam_size):
    with torch.no_grad():

        true_outputs = []
        decoder_outputs = []

        for i, (inp, inp_lens, output, out_mask, out_max) in enumerate(data_loader):
            if i * batch_size >= 10000:
                break
            inp = inp.transpose(0,1).to(device)
            output = output.transpose(0,1).to(device)

            true_outputs.append([[str(tok.item()) for tok in out if tok != 0] for out in output])
            encoder_hidden = encoder.initHidden()
            input_len = inp.shape[1]

            encoder_outputs = torch.zeros(input_len, batch_size, 1, HIDDEN_SIZE, device=device)
            encoder_hiddens = torch.zeros(input_len, 1, batch_size, HIDDEN_SIZE, device=device)

            encoder_hidden = encoder.initHidden()
            # Encode
            for ec_idx in range(input_len):
                # input batch_size * 1
                encoder_output, encoder_hidden = encoder(inp[:, ec_idx].unsqueeze(1), encoder_hidden)
                encoder_outputs[ec_idx] = encoder_output
                encoder_hiddens[ec_idx] = encoder_hidden

            # Decode
            decoder_input = torch.tensor([SOS_ID] * batch_size, device=device)
            decoder_hidden = encoder_hidden

            # BEAM SEARCH BELOW
            # candidates are stored as : (curr_scores, curr_sequences, decoder_hidden, decoder_input)
            candidates = [([0] * batch_size, [[str(SOS_ID)] * batch_size], decoder_hidden, decoder_input) for _ in range(beam_size)]
            for t in range(out_max):
                next_candidates = []
                next_candidate_scores = [SortedList() for _ in range(batch_size)] # list of sorted lists of candidate scores for each sentence
                next_candidate_inputs = [[] * batch_size] # dict from total curr score to next candidate token for each sentence
                next_candidate_seqs = [[] * batch_size] # dict from total curr score to next candidate sequence
                for curr_scores, curr_seqs, decoder_hidden, decoder_input in candidates:
                    # generate output + next hidden state given input and current hidden state of the candidate
                    decoder_output, decoder_hidden, _ = self.decoder(decoder_input.unsqueeze(1), decoder_hidden)
                    decoder_output = decoder_output.squeeze(1).to(device) # get rid of the seq dimention
                    topv, topi = decoder_output.topk(beam_size)
                    for k in range(beam_size): 
                        # calculate for each sentence the next `beam_size` best possible next tokens
                        for i in range(len(topi)):
                            if len(topi[i]) != beam_size:
                                print('uhh', i, len(topi[i]))
                        decoder_input = torch.LongTensor([topi[i][k] for i in range(batch_size)]).to(device)
                        for i in range(batch_size):
                            # for sentence `i`, add `topi[i][k]` as a candidate if the new score for the seq is better than the other candidates
                            if (curr_seqs[i][-1] == EOS_ID): #don't do anything with current sequence if already is EOS
                                continue 
                            curr_score = curr_scores[i] + topv[i][k].item()
                            if (len(next_candidate_scores[i]) < beam_size or curr_score < next_candidate_scores[i][beam_size - 1]) and curr_score not in next_candidate_scores[i]:
                                if len(next_candidate_scores[i]) == beam_size:
                                    next_candidate_inputs[i] = [candidate_input for candidate_input in next_candidate_inputs[i] if candidate_input[0] != next_candidate_scores[i][beam_size - 1]] # delete candidate associated with score
                                    next_candidate_seqs[i] = [candidate_seq for candidate_seq in next_candidate_seqs[i] if candidate_seq[0] != next_candidate_scores[i][beam_size - 1]] # delete candidate associated with score
                                    del next_candidate_scores[i][beam_size - 1] # delete associated score
                                next_candidate_scores[i].insort(curr_score) # insert new score in sorted order to scores lists for the i'th sentence
                                next_candidate_inputs[i].append((curr_score, topi[i][k].item())) # insert new token value for score key for the i'th sentence
                                next_candidate_seqs[i].append((curr_score, curr_seqs[i] + [str(topi[i][k].item())]))
                next_candidate_scores = [[score for score in next_candidate_scores[i]] for i in range(batch_size)]
                next_candidate_seqs = [[candidate_seq[1] for candidate_seq in sorted(next_candidate_seqs[i])] for i in range(batch_size)]
                next_candidate_inputs = [[candidate_input[1] for candidate_input in sorted(next_candidate_inputs[i])] for i in range(batch_size)]
#                 now that each best 3 sequences for each sentence is selected, create new candidates.
                for k in range(min(len(next_candidate_inputs[0]), len(next_candidate_scores[0]), len(next_candidate_seqs[0]))):
                    decoder_input = torch.LongTensor([[next_candidate_inputs[i][k] for i in range(batch_size)]])
                    next_scores = [next_candidate_scores[i][k] for i in range(batch_size)]
                    next_seqs = [next_candidate_seqs[i][k] for i in range(batch_size)]
                    next_candidates.append((next_scores, next_seqs, decoder_hidden, decoder_input))                            
                candidates = next_candidates
            pred_outputs = [pred_out + [str(EOS_ID)] for pred_out in candidates[0][1]]
            decoder_outputs.append(pred_outputs)
            print(' '.join([en.id2word[int(i)] for i in pred_outputs[0]])) 
            print(candidates[0][0])

    return corpus_bleu(decoder_outputs, true_outputs, 4)

def bleuScoreAttention(data_loader, encoder, decoder, batch_size, HIDDEN_SIZE, beam_size):
    with torch.no_grad():

        true_outputs = []
        decoder_outputs = []

        for i, (inp, inp_lens, output, out_mask, out_max) in enumerate(data_loader):
            if i * batch_size >= 10000 or len(inp[0]) != batch_size:
                break
            inp = inp.transpose(0,1).to(device)
            output = output.transpose(0,1).to(device)

            true_outputs.append([[str(tok.item()) for tok in out if tok != 0] for out in output])
            encoder_hidden = encoder.initHidden()
            input_len = inp.shape[1]

            encoder_outputs = torch.zeros(input_len, batch_size, 1, HIDDEN_SIZE, device=device)
            encoder_hiddens = torch.zeros(input_len, 1, batch_size, HIDDEN_SIZE, device=device)

            encoder_hidden = encoder.initHidden()
            # Encode
            for ec_idx in range(input_len):
                # input batch_size * 1
                encoder_output, encoder_hidden = encoder(inp[:, ec_idx].unsqueeze(1), encoder_hidden)
                encoder_outputs[ec_idx] = encoder_output
                encoder_hiddens[ec_idx] = encoder_hidden

            # Decode
            decoder_input = torch.tensor([SOS_ID] * batch_size, device=device)
            decoder_hidden = encoder_hidden

            # BEAM SEARCH BELOW
            # candidates are stored as : (curr_scores, curr_sequences, decoder_hidden, decoder_input)
            candidates = [([0 for _ in range(batch_size)], [[str(SOS_ID)] for _ in range(batch_size)], decoder_hidden, decoder_input) for _ in range(beam_size)]
            for t in range(out_max):
                #print(t)
                next_candidates = []
                next_candidate_scores = [SortedList() for _ in range(batch_size)] # list of sorted lists of candidate scores for each sentence
                next_candidate_inputs = [[] for _ in range(batch_size)] # dict from total curr score to next candidate token for each sentence
                next_candidate_seqs = [[] for _ in range(batch_size)] # dict from total curr score to next candidate sequence
                for curr_scores, curr_seqs, decoder_hidden, decoder_input in candidates:
                    # generate output + next hidden state given input and current hidden state of the candidate
                    #print(np.shape(decoder_input.unsqueeze(1)), np.shape(decoder_hidden), np.shape(encoder_hiddens.squeeze(1)))
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input.unsqueeze(1), decoder_hidden, encoder_hiddens.squeeze(1))
                    decoder_output = decoder_output.squeeze(1).to(device) # get rid of the seq dimention
                    topv, topi = decoder_output.topk(beam_size)
                    for k in range(beam_size): 
                        # calculate for each sentence the next `beam_size` best possible next tokens
                        for i in range(len(topi)):
                            if len(topi[i]) != beam_size:
                                print('uhh', i, len(topi[i]))
                        decoder_input = torch.LongTensor([topi[i][k] for i in range(batch_size)]).to(device)
                        for i in range(batch_size):
                            # for sentence `i`, add `topi[i][k]` as a candidate if the new score for the seq is better than the other candidates
                            if (curr_seqs[i][-1] == EOS_ID): #don't do anything with current sequence if already is EOS
                                continue 
                            curr_score = curr_scores[i] + topv[i][k].item()
                            if (len(next_candidate_scores[i]) < beam_size or curr_score < next_candidate_scores[i][beam_size - 1]) and curr_score not in next_candidate_scores[i]:
                                if len(next_candidate_scores[i]) == beam_size:
                                    next_candidate_inputs[i] = [candidate_input for candidate_input in next_candidate_inputs[i] if candidate_input[0] != next_candidate_scores[i][beam_size - 1]] # delete candidate associated with score
                                    next_candidate_seqs[i] = [candidate_seq for candidate_seq in next_candidate_seqs[i] if candidate_seq[0] != next_candidate_scores[i][beam_size - 1]] # delete candidate associated with score
                                    del next_candidate_scores[i][beam_size - 1] # delete associated score
                                next_candidate_scores[i].insort(curr_score) # insert new score in sorted order to scores lists for the i'th sentence
                                next_candidate_inputs[i].append((curr_score, topi[i][k].item())) # insert new token value for score key for the i'th sentence
                                next_candidate_seqs[i].append((curr_score, curr_seqs[i] + [str(topi[i][k].item())]))
                next_candidate_scores = [[score for score in next_candidate_scores[i]] for i in range(batch_size)]
                next_candidate_seqs = [[candidate_seq[1] for candidate_seq in sorted(next_candidate_seqs[i])] for i in range(batch_size)]
                next_candidate_inputs = [[candidate_input[1] for candidate_input in sorted(next_candidate_inputs[i])] for i in range(batch_size)]
#                 now that each best 3 sequences for each sentence is selected, create new candidates.
                for k in range(min(len(next_candidate_inputs[0]), len(next_candidate_scores[0]), len(next_candidate_seqs[0]))):
                    decoder_input = torch.LongTensor([next_candidate_inputs[i][k] for i in range(batch_size)])
                    next_scores = [next_candidate_scores[i][k] for i in range(batch_size)]
                    next_seqs = [next_candidate_seqs[i][k] for i in range(batch_size)]
                    next_candidates.append((next_scores, next_seqs, decoder_hidden, decoder_input))                            
                candidates = next_candidates
            pred_outputs = [pred_out + [str(EOS_ID)] for pred_out in candidates[0][1]]
            decoder_outputs += pred_outputs
            #print(' '.join([en.id2word[int(i)] for i in pred_outputs[0]])) 
            #print(candidates[0][0])
    predict = []
    for seq in decoder_outputs:
        seq_toks = []
        for tok in seq:
            seq_toks.append(tok)
            if tok == '3':
                break
        predict.append(seq_toks)
    print(np.shape(predict), np.shape(true_outputs))
    return corpus_bleu(predict, true_outputs, 4)



def bleuScoreConv(data_loader, encoder, decoder, batch_size, HIDDEN_SIZE, beam_size):
    with torch.no_grad():

        true_outputs = []
        decoder_outputs = []

        for i, (inp, inp_lens, output, out_mask, out_max) in enumerate(data_loader):
            if i * batch_size >= 10000 or len(inp[0]) != batch_size:
                break
            inp = inp.transpose(0,1).to(device)
            output = output.transpose(0,1).to(device)

            true_outputs.append([[str(tok.item()) for tok in out if tok != 0] for out in output])
            input_len = inp.shape[1]

            encoder_outputs = torch.zeros(input_len, batch_size, 1, HIDDEN_SIZE, device=device)
            encoder_hiddens = torch.zeros(input_len, 1, batch_size, HIDDEN_SIZE, device=device)
            encoder_hidden = None
            position_ids = torch.LongTensor(range(0, input_len)).to(device)
            encoder_conv_out = encoder(position_ids, inp)
            encoder_hiddens = encoder_conv_out.unsqueeze(1).transpose(0, 3).transpose(2, 3)

            # Decode
            decoder_input = torch.tensor([SOS_ID] * batch_size, device=device)
            decoder_hidden = decoder.initHidden()

            # BEAM SEARCH BELOW
            # candidates are stored as : (curr_scores, curr_sequences, decoder_hidden, decoder_input)
            candidates = [([0 for _ in range(batch_size)], [[str(SOS_ID)] for _ in range(batch_size)], decoder_hidden, decoder_input) for _ in range(beam_size)]
            for t in range(out_max):
                next_candidates = []
                next_candidate_scores = [SortedList() for _ in range(batch_size)] # list of sorted lists of candidate scores for each sentence
                next_candidate_inputs = [[] for _ in range(batch_size)] # dict from total curr score to next candidate token for each sentence
                next_candidate_seqs = [[] for _ in range(batch_size)] # dict from total curr score to next candidate sequence
                for curr_scores, curr_seqs, decoder_hidden, decoder_input in candidates:
                    # generate output + next hidden state given input and current hidden state of the candidate
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input.unsqueeze(1), decoder_hidden, encoder_hiddens.squeeze(1))
                    decoder_output = decoder_output.squeeze(1).to(device) # get rid of the seq dimention
                    topv, topi = decoder_output.topk(beam_size)
                    for k in range(beam_size): 
                        # calculate for each sentence the next `beam_size` best possible next tokens
                        for i in range(len(topi)):
                            if len(topi[i]) != beam_size:
                                print('uhh', i, len(topi[i]))
                        decoder_input = torch.LongTensor([topi[i][k] for i in range(batch_size)]).to(device)
                        for i in range(batch_size):
                            # for sentence `i`, add `topi[i][k]` as a candidate if the new score for the seq is better than the other candidates
                            if (curr_seqs[i][-1] == EOS_ID): #don't do anything with current sequence if already is EOS
                                continue 
                            curr_score = curr_scores[i] + topv[i][k].item()
                            if (len(next_candidate_scores[i]) < beam_size or curr_score < next_candidate_scores[i][beam_size - 1]) and curr_score not in next_candidate_scores[i]:
                                if len(next_candidate_scores[i]) == beam_size:
                                    next_candidate_inputs[i] = [candidate_input for candidate_input in next_candidate_inputs[i] if candidate_input[0] != next_candidate_scores[i][beam_size - 1]] # delete candidate associated with score
                                    next_candidate_seqs[i] = [candidate_seq for candidate_seq in next_candidate_seqs[i] if candidate_seq[0] != next_candidate_scores[i][beam_size - 1]] # delete candidate associated with score
                                    del next_candidate_scores[i][beam_size - 1] # delete associated score
                                next_candidate_scores[i].insort(curr_score) # insert new score in sorted order to scores lists for the i'th sentence
                                next_candidate_inputs[i].append((curr_score, topi[i][k].item())) # insert new token value for score key for the i'th sentence
                                next_candidate_seqs[i].append((curr_score, curr_seqs[i] + [str(topi[i][k].item())]))
                next_candidate_scores = [[score for score in next_candidate_scores[i]] for i in range(batch_size)]
                next_candidate_seqs = [[candidate_seq[1] for candidate_seq in sorted(next_candidate_seqs[i])] for i in range(batch_size)]
                next_candidate_inputs = [[candidate_input[1] for candidate_input in sorted(next_candidate_inputs[i])] for i in range(batch_size)]
#                 now that each best 3 sequences for each sentence is selected, create new candidates.
                for k in range(min(len(next_candidate_inputs[0]), len(next_candidate_scores[0]), len(next_candidate_seqs[0]))):
                    decoder_input = torch.LongTensor([next_candidate_inputs[i][k] for i in range(batch_size)])
                    next_scores = [next_candidate_scores[i][k] for i in range(batch_size)]
                    next_seqs = [next_candidate_seqs[i][k] for i in range(batch_size)]
                    next_candidates.append((next_scores, next_seqs, decoder_hidden, decoder_input))                            
                candidates = next_candidates
            pred_outputs = [pred_out + [str(EOS_ID)] for pred_out in candidates[0][1]]
            decoder_outputs += pred_outputs
            #print(' '.join([en.id2word[int(i)] for i in pred_outputs[0]])) 
            #print(candidates[0][0])
    predict = []
    for seq in np.array(candidates[0][1]).T.astype(str):
        seq_toks = []
        for tok in seq:
            seq_toks.append(tok)
            if tok == '3':
                break
        predict.append(seq_toks)
    return corpus_bleu(decoder_outputs, true_outputs, 4)
