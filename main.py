#############################################
#This script includes a number of functions 
#for measuring the behavior of a feed-forward 
#LSTM with two layers.
#Use case in mind was norming human 
#for easy comparision to models
#############################################
import math
import glob
import sys
import warnings 
warnings.filterwarnings("ignore") #wild
import torch
import torch.nn as nn
import data_test
import rsa_data as data
import model as m

#set device to cpu for work on desktop :/
device = torch.device('cpu')

#set loss function to be cross entropy
criterion = nn.CrossEntropyLoss()

#### Information theory stuff ###
def get_entropy(state):
    ''' Compute entropy of input vector '''
    # requirments: state should be a vector scoring possible classes
    # returns scalar

    #get probabilities of outputs
    probs = nn.functional.softmax(state, dim=0)
    #get log probs
    logprobs = nn.functional.log_softmax(state, dim=0)
    #entropy is sum over i, -prob(i)*logprob(i)
    prod = probs.data * logprobs.data
    #set nans to zero
    prod[prod != prod] = 0

    return torch.Tensor([-1 * torch.sum(prod)]).to(device)

def get_surps(state):
    ''' Compute surprisal for each element in vector '''
    #Get log prob of softmax layer, so this will be vector
    #of surprisals for given output
    logprobs = nn.functional.log_softmax(state, dim=0)
    return -1 * logprobs

#return word, surp, H for each word in state
def get_IT(state, obs, corpus):

    metrics = []
    #get entropies and put into shannons 
    Hs = torch.log2(torch.exp(torch.squeeze(apply(get_entropy, state))))
    #get surprisals and put into shannons 
    surps = torch.log2(torch.exp(apply(get_surps, state)))

    for corpuspos, targ in enumerate(obs):
        #get word 
        word = corpus.dictionary.idx2word[int(targ)]
        if word == "<eos>":
            #skip over EOS
            continue
        #Get surprisal of target at time step 
        surp = float(surps[corpuspos][int(targ)].data)
        #Get entropy at time step
        H = float(Hs[corpuspos])
        metrics.append((word, surp, H))
    return metrics

#other helpers
def test_get_batch(source):
    ''' Creates an input/target pair for evaluation '''
    seq_len = len(source) - 1
    #Get all words except last one
    data = source[:seq_len]
    #Get all targets
    target = source[1:1+seq_len].view(-1)
    return data, target

def repackage_hidden(in_state):
    if isinstance(in_state, torch.Tensor):
        return in_state.detach()
    else:
        return tuple(repackage_hidden(value) for value in in_state)

def apply(func, apply_dimension):
    output_list = [func(m) for m in torch.unbind(apply_dimension, dim=0)]
    return torch.stack(output_list, dim=0)

def test_IT(data_source, corpus, model):
    ''' Given a list of one hot encoded data, return information theoretic measures '''

    model.eval()

    total_loss = 0.

    #Get vocab size for beam
    ntokens = len(corpus.dictionary)

    #For each sent
    values = []
    for i in range(len(data_source)):
        sent_ids = data_source[i].to(device)
        hidden = model.init_hidden(1)

        data, targets = test_get_batch(sent_ids)

        data = data.unsqueeze(1)

        output, hidden = model(data, hidden)

        output_flat = output.view(-1, ntokens)
        metrics = get_IT(output_flat, targets, corpus)
        values.append(metrics)
    return values

def find_det_nouns(metrics):

    period_idx = []
    for x in range(len(metrics)):
        if metrics[x][0] == '.':
            period_idx.append(x)

    context_det = metrics[period_idx[0]-2]
    context_noun = metrics[period_idx[0]-1]

    target_det = metrics[period_idx[1]-2]
    target_noun = metrics[period_idx[1]-1]

    if context_det[1] > target_det[1]:
        print('fudge')
        print(metrics)
        print(context_det, context_noun)
        print(target_det, target_noun)

def run_norming(stim_file, vocab_file, model_files, header=False, 
        multisent_flag = False, filter_file = None, verbose=False):
    ''' Given a stimuli file, model vocabulary file and model files
    return information about frequency and information
    theoretic measures'''

    #hard code data_dir
    data_path = './'


    #set loss function to be cross entropy
    criterion = nn.CrossEntropyLoss()

    #Load experiments
    #__iter__ is over pairs of Min and Sub verbs
    #includes RSA results by model (ie by participant)
    EXP = data.Stim(stim_file, header, filter_file)

    #Loop through the models
    for model_file in model_files:
        if verbose:
            print('testing model:', model_file)

        #load the model
        with open(model_file, 'rb') as f:
            #run on local cpu for now
            model = torch.load(f, map_location='cpu')

            # make in continous chunk of memory for speed
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            model.rnn.flatten_parameters()

            #Check we have the correct version
            try:
                hidden = model.init_hidden(1)

                test_data = torch.tensor([0]).unsqueeze(0)

                output, hidden = model(data, hidden)
            #Problem with diff versions of torch
            except:
                new_model = m.RNNModel('LSTM', 28439, 400, 400, 2, None, 0.2, tie_weights=True).to(device)
                new_model.load_state_dict(model.state_dict())
                model = new_model

        #loop through experimental items for EXP
        for x in range(len(EXP.UNK_SENTS)):
            sentences = list(EXP.UNK_SENTS[x])

            #Create corpus wrapper (this is for one hoting data)
            corpus = data_test.TestSent(data_path, vocab_file, 
                    sentences, multisent_flag)
            #Get one hots
            sent_ids = corpus.get_data()

            values = test_IT(sent_ids, corpus, model)

            EXP.load_IT(model_file, x, values, multisent_flag)
            break

    return EXP

stim_file = 'stimuli/RSA_Analysis.xlsx'
vocab_file = 'models/vocab'
model_files = glob.glob('models/*.pt')[:1]

header = True
multisent_flag = True
filter_file = None
#filter_file = 'filter'
verbose = True
hasSim = False
only_avg = True

EXP = run_norming(stim_file, vocab_file, model_files, header, multisent_flag, filter_file, verbose)
EXP.save_csv('pilot_'+stim_file.split('/')[-1], model_files, only_avg, hasSim)
