import sys
import warnings
import torch
import torch.nn as nn
import model
import numpy as np
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='Matrix of Similarities from LSTM Language Model Embeddings')

parser.add_argument('--model_file', type=str, default='models/LSTM_400_80m_a_10-d0.2.pt',
                    help='model to run')

parser.add_argument('--vocab_file', type=str, default='models/vocab',
                    help='vocab file')

parser.add_argument('--word_file', type=str, 
        default='stimuli/words.xlsx', 
        help='path to stimuli file')

parser.add_argument('--output_file', type=str, 
        default='results/embedding_matrix.xlsx', 
        help='Ouput file name')

args = parser.parse_args()

#for setting device
cuda = False
device = torch.device("cuda" if cuda else "cpu")

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
target_words = EXP[0].values
target_set = set(target_words)



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

embeddings_norm = np.linalg.norm(EMBEDDINGS, axis=1)


data = []
for x, word in enumerate(target_words):
    if word not in word2idx:
        word = "<unk>"
        target_words[x] = word

    target_embedding = EMBEDDINGS[word2idx[word]]
    target_norm = embeddings_norm[word2idx[word]]
    num = np.dot(EMBEDDINGS, target_embedding)
    denom = target_norm*embeddings_norm
    sim = list(num/denom)
    data.append(sim)

data = np.asarray(data).T
dataframe = pd.DataFrame(data, columns = target_words) 
dataframe.insert(0, "word1", WORDS)
dataframe.to_excel(args.output_file, index=False)





'''
num = np.einsum('ij, kj->ik', target_embeddings, EMBEDDINGS)
denom = np.einsum('i,j', target_embeddings_norm, embeddings_norm)
sims = num/denom

print(sims)
'''

'''
for word in target_words:
    if word not in word2idx:
        word = '<unk>'
    print(word)
    target_idx = word2idx[word]
    print(target_idx)
    target_embed = EMBEDDINGS[target_idx]
    target_norm = np.linalg.norm(target_embed)
    print(target_norm)
    denom = target_norm*embeddings_norm
    print(denom)
'''

'''
#Using einsum to make it faster. I don't often go for one liners, but when 
#I do I make sure nobody can understand them lol
num = np.einsum('ij, kj->ik', EMBEDDINGS, EMBEDDINGS)
denom = np.einsum('i,j', embeddings_norm, embeddings_norm)
sims = num/denom

#results = np.vstack((['word1']+WORDS, np.column_stack([WORDS, sims])))
#print(results)
#np.savetxt(outfile_name, sims, delimiter=',')
dataframe = pd.DataFrame(sims, columns = WORDS) 
dataframe.insert(0, "word1", WORDS)
dataframe.to_csv(outfile_name, index=False)
'''

