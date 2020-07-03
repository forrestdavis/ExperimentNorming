#############################################
#This script includes a number of functions 
#for measuring the behavior of a feed-forward 
#LSTM with two layers.
#Added in July: representational analysis 
#Use case in mind was norming human 
#for easy comparision to models
#Created by Forrest Davis 
#############################################
import math
import glob
import sys
import warnings 
warnings.filterwarnings("ignore") #wild
import torch
import torch.nn as nn
import data_test
import data
import model as m
import numpy as np
import pandas as pd

#set device to cpu if working on laptop :)
device = torch.device('cpu')
#set device to cpu if working on desktop :))))
#device = torch.device("cuda:0")

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

def get_sims(target_ids, sent_ids, corpus, model):

    #cancel dropout
    model.eval()

    #don't learn
    model.zero_grad()

    target_ids = target_ids[0][:-1].to(device)
    target_data, target_targets = test_get_batch(target_ids)
    target_data = target_data.unsqueeze(1)

    hidden = model.init_hidden(1)

    output, hidden = model(target_data, hidden)

    #hidden[0] is hidden; hidden[1] is cell state
    hidden = hidden[0].data
    layer_1_target = hidden[1].cpu().squeeze()

    SIMS = []
    
    for i in range(len(sent_ids)):
        model.zero_grad()
        sims = []
        s_ids = sent_ids[i]
        data, targets = test_get_batch(s_ids)

        data = data.unsqueeze(1)

        hidden = model.init_hidden(1)

        for word_index in range(data.size(0)):
            hidden = repackage_hidden(hidden)

            word_input = data[word_index]
            
            #What's going in 
            input_word = corpus.dictionary.idx2word[int(word_input.data)]
            output, hidden = model(torch.tensor([[word_input]]).to(device), hidden)
            if input_word == "<eos>":
                continue

            h = hidden[0].data
            layer_1 = h[1].cpu().squeeze()

            sim = np.corrcoef(layer_1_target, layer_1)[0, 1]
            sims.append((input_word, sim))
        SIMS.append(sims)

    return SIMS

def load_sents(test_file, hasHeader=True):

    if hasHeader:
        EXP = pd.read_excel(stim_file)
    else:
        EXP = pd.read_excel(stim_file, header=None)

    header = EXP.columns.values

    sents = []
    for x in [0, 2]:
        for y in EXP[header[x]]:
            sents.append(y)

    return sents

def load_model(model_file):
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

    return model


def adapt(test_file, vocab_file, model_files):

    #hard code data_dir
    data_path = './'


    #set loss function to be cross entropy
    criterion = nn.CrossEntropyLoss()

    #Load sents
    sents = load_sents(test_file)

    #backprop during eval
    torch.backends.cudnn.enabled = False

    #Loop through the models
    for model_file in model_files:
        print('testing model:', model_file)

        #Create corpus wrapper (this is for one hoting data)
        corpus = data_test.TestSent(data_path, vocab_file, 
                sents, False)
        #Get one hots
        sent_ids = corpus.get_data()
        for i in range(len(sent_ids)):

            #load model 
            model = load_model(model_file)
            model.eval()

            sent_id = sent_ids[i].to(device)

            #Pre-adapt ITs
            hidden = model.init_hidden(1)
            data, targets = test_get_batch(sent_id)

            data = data.unsqueeze(1)
            output, hidden = model(data, hidden)

            output_flat = output.view(-1, len(corpus.dictionary))
            pre_metrics = get_IT(output_flat, targets, corpus)
            #print(pre_metrics)

            #backprop
            loss = criterion(output_flat, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            for param in model.parameters():
                if param.grad is not None:
                    param.data.add_(-20, param.grad.data)

            #Post-adapt ITs
            hidden = model.init_hidden(1)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, len(corpus.dictionary))
            post_metrics = get_IT(output_flat, targets, corpus)
            #print(post_metrics)

            out = []
            for x in range(len(post_metrics)):
                pre_metric = pre_metrics[x]
                post_metric = post_metrics[x]

                word = pre_metric[0]

                assert word == post_metric[0]
                pre_surp = str(pre_metric[1])
                post_surp = str(post_metric[1])
                out += [word, pre_surp, post_surp]

            out.append('delta')
            out.append(str(float(out[-3])-float(out[-2])))

            print(sents[i] + ',' + ','.join(out))

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

    return EXP

def run_RSA(stim_file, vocab_file, model_files, header=False, 
        multisent_flag = False, filter_file = None, verbose=False):

    ''' Given a stimuli file, model vocabulary file and model files
    return information about information
    theoretic measures and similarity'''

    #hard code data_dir
    data_path = './'


    #set loss function to be cross entropy
    criterion = nn.CrossEntropyLoss()

    #Load experiments
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

            target = sentences[:1]
            sentences = sentences[1:]

            #Create corpus wrapper (this is for one hoting data)
            corpus = data_test.TestSent(data_path, vocab_file, 
                    target, False)
            #Get one hots
            target_ids = corpus.get_data()

            #Create corpus wrapper (this is for one hoting data)
            corpus = data_test.TestSent(data_path, vocab_file, 
                    sentences, multisent_flag)
            #Get one hots
            sent_ids = corpus.get_data()

            sims = get_sims(target_ids, sent_ids, corpus, model)

            values = test_IT(sent_ids, corpus, model)

            EXP.load_IT(model_file, x, values, multisent_flag, sims)

    return EXP

stim_file = 'stimuli/Book1.xlsx'
vocab_file = 'models/vocab' 
model_files = glob.glob('models/*.pt')[:1]
adapt(stim_file, vocab_file, model_files)
'''
stim_file = 'stimuli/RSA_Analysis.xlsx'
vocab_file = 'models/vocab'
model_files = glob.glob('models/*.pt')[:1]

header = True
multisent_flag = True
filter_file = None
#filter_file = 'filter'
verbose = True
hasSim = True
only_avg = True

#EXP = run_norming(stim_file, vocab_file, model_files, header, multisent_flag, filter_file, verbose)
EXP = run_RSA(stim_file, vocab_file, model_files, header, multisent_flag, filter_file, verbose)
#EXP.save_csv('pilot_UNMULTI_'+stim_file.split('/')[-1], model_files, only_avg, hasSim)
'''
