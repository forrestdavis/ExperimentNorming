import sys
import warnings
import torch
import torch.nn as nn
import model
import numpy as np
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='Across Stimuli Similarities from LSTM Language Model Embeddings')

parser.add_argument('--model_file', type=str, default='models/LSTM_400_80m_a_10-d0.2.pt',
                    help='model to run')

parser.add_argument('--vocab_file', type=str, default='models/vocab',
                    help='vocab file')

parser.add_argument('--word_file', type=str, 
        default='stimuli/static_stimuli.xlsx', 
        help='path to stimuli file')

parser.add_argument('--output_file', type=str, 
        default='results/static_similarities.xlsx', 
        help='Ouput file name')

args = parser.parse_args()

#for setting device
cuda = False
device = torch.device("cuda" if cuda else "cpu")

def cos_sim(a, b):

    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

#Read in stimuli files as pandas dataframe
def read_stim_file(stim_file, hasHeader=False):

    if hasHeader:
        EXP = pd.read_excel(stim_file)
    else:
        EXP = pd.read_excel(stim_file, header=None)

    header = EXP.columns.values

    return EXP, header

###############################################################################
# Load target words
###############################################################################
EXP, _ = read_stim_file(args.word_file)
baseline_words = EXP[0].values
target_sents = EXP[1].values
print(baseline_words, target_sents)
#target_set = set(target_words)



###############################################################################
# Load the vocab
###############################################################################
idx2word = {}
word2idx = {}
with open(args.vocab_file, 'r') as f:
    idx = 0
    for line in f:
        line = line.strip()
        idx2word[idx] = line
        word2idx[line] = idx
        idx += 1

###############################################################################
# Load the model
###############################################################################

with open(args.model_file, 'rb') as f:
    if cuda:
        model = torch.load(f).to(device)
    else:
        model = torch.load(f, map_location='cpu')

    if cuda and (not args.single) and (torch.cuda.device_count() > 1):
        # If applicable, use multi-gpu for training
        # Scatters minibatches (in dim=1) across available GPUs
        model = nn.DataParallel(model, dim=1)
    if isinstance(model, torch.nn.DataParallel):
        # if multi-gpu, access real model for training
        model = model.module
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

###############################################################################
# Get Embeddings
###############################################################################

#word, embedding
WORDS = []
EMBEDDINGS = []

#print(model.encoder(torch.LongTensor([w for w in range(model.encoder.num_embeddings)])))
for idx, embed in enumerate(model.encoder(torch.LongTensor([w for w in range(model.encoder.num_embeddings)])).data.numpy().tolist()):
    word = idx2word[idx]
    if word == ',':
        word = "COMMA"
    WORDS.append(word)
    EMBEDDINGS.append(embed)

    #print(word+' '+' '.join(str(f) for f in embed))

data = []
for x, word in enumerate(baseline_words):
    if word not in word2idx:
        word = "<unk>"
        baseline_words[x] = word

    baseline_embedding = EMBEDDINGS[word2idx[word]]
    row = []

    target_sent = target_sents[x]
    target_words = target_sent.split(' ')
    for y, target_word in enumerate(target_words):
        if target_word not in word2idx:
            target_word = "<unk>"
            target_words[y] = target_word
        target_embedding = EMBEDDINGS[word2idx[target_word]]
        sim = cos_sim(baseline_embedding, target_embedding)
        row.append(sim)
        target_sents[x] = ' '.join(target_words)
    data.append(row)

row_header = []
for i in range(len(target_sents[0].split(' '))):
    row_header.append('word'+str(i))

data = np.asarray(data)
dataframe = pd.DataFrame(data, columns = row_header) 
dataframe.insert(0, "baseline", baseline_words)
dataframe.insert(1, "stimuli", target_sents)
dataframe.to_excel(args.output_file, index=False)
