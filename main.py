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
#device = torch.device('cpu')
#set device to cpu if working on desktop :))))
device = torch.device("cuda:0")

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

    target_ids = target_ids[0].to(device)
    target_data, target_targets = test_get_batch(target_ids)
    target_data = target_data.unsqueeze(1)

    hidden = model.init_hidden(1)

    output, hidden = model(target_data, hidden)

    #hidden[0] is hidden; hidden[1] is cell state
    hidden = hidden[0].data
    layer_1_target = hidden[-1].cpu().squeeze()

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

def load_RSA_adapt(stim_file, hasHeader=True):

    if hasHeader:
        EXP = pd.read_excel(stim_file)
    else:
        EXP = pd.read_excel(stim_file, header=None)

    header = EXP.columns.values

    baselines = []
    comps = []
    sents = []
    ents = []
    break_point = len(EXP[header[0]])
    for x in range(len(header)):
        for y in EXP[header[x]]:
            if x == 0 or x == 4:
                baselines.append(y)
            elif x == 1 or x == 5:
                comps.append(y)
            elif x == 2 or x == 6:
                sents.append(y)
            else:
                ents.append(y)

    return baselines, comps, sents, ents, break_point

def load_adapt(stim_file, hasHeader=True):

    if hasHeader:
        EXP = pd.read_excel(stim_file)
    else:
        EXP = pd.read_excel(stim_file, header=None)

    header = EXP.columns.values

    sents = []
    ents = []
    break_point = len(EXP[header[0]])
    for x in range(len(header)):
        for y in EXP[header[x]]:
            if x == 0 or x == 2:
                sents.append(y)
            else:
                ents.append(y)

    return sents, ents, break_point

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
            try:
                new_model = m.RNNModel('LSTM', 28439, 400, 400, 2, None, 0.2, tie_weights=True).to(device)
                new_model.load_state_dict(model.state_dict())
                model = new_model
            except:
                new_model = m.RNNModel('LSTM', 50002, 400, 400, 2, None, 0.2, tie_weights=True).to(device)
                new_model.load_state_dict(model.state_dict())
                model = new_model

    return model

def RSA_adapt(test_file, vocab_file, model_files, has_header):

    #hard code data_dir
    data_path = './'

    #set loss function to be cross entropy
    criterion = nn.CrossEntropyLoss()

    baselines, comps, sents, ents, break_point = load_RSA_adapt(test_file, 
                                                        has_header)

    #backprop during eval
    torch.backends.cudnn.enabled = False

    # models X sents
    RSAS = {}

    #Loop through the models
    for model_file in model_files:
        print('testing model:', model_file)

        if model_file not in RSAS:
            RSAS[model_file] = []

        #Create corpus wrapper (this is for one hoting data)
        corpus = data_test.TestSent(data_path, vocab_file, 
                sents, False)
        #Get one hots
        sent_ids = corpus.get_data()

        #Create corpus wrapper (this is for one hoting data)
        corpus = data_test.TestSent(data_path, vocab_file, 
                baselines, False)
        #Get one hots
        base_ids = corpus.get_data()

        #Create corpus wrapper (this is for one hoting data)
        corpus = data_test.TestSent(data_path, vocab_file, 
                comps, False)
        #Get one hots
        comp_ids = corpus.get_data()
        
        for i in range(len(sent_ids)):

            #load model 
            model = load_model(model_file)
            model.eval()

            sent_id = sent_ids[i].to(device)
            base_id = base_ids[i].to(device)
            comp_id = comp_ids[i].to(device)

            #Pre-adapt sim
            sims = get_sims([base_id], [comp_id], corpus, model)
            pre_sim = sims[-1][-1][-1]
            #print(pre_sim)

            #Adapt
            hidden = model.init_hidden(1)
            data, targets = test_get_batch(sent_id)

            data = data.unsqueeze(1)
            output, hidden = model(data, hidden)

            output_flat = output.view(-1, len(corpus.dictionary))

            #backprop
            loss = criterion(output_flat, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            for param in model.parameters():
                if param.grad is not None:
                    param.data.add_(-20, param.grad.data)

            #Get post sim
            sims = get_sims([base_id], [comp_id], corpus, model)
            post_sim = sims[-1][-1][-1]

            RSAS[model_file].append((pre_sim, post_sim))

    return RSAS, baselines, comps, sents, ents, break_point

def adapt(test_file, vocab_file, model_files, has_header):

    #hard code data_dir
    data_path = './'


    #set loss function to be cross entropy
    criterion = nn.CrossEntropyLoss()

    #Load sents
    sents, ents, break_point = load_adapt(test_file, has_header)

    #backprop during eval
    torch.backends.cudnn.enabled = False

    # models X sents
    DELTAS = {}

    #Loop through the models
    for model_file in model_files:
        print('testing model:', model_file)

        if model_file not in DELTAS:
            DELTAS[model_file] = []

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

            delta = get_delta(pre_metrics, post_metrics)

            DELTAS[model_file].append(delta)

    return DELTAS, sents, ents, break_point

def get_delta(pre_metrics, post_metrics):

    return pre_metrics[-1][1] - post_metrics[-1][1]

def run_RSA_adapt(test_file, vocab_file, model_files,  
        out_name, has_header = True, only_avg = False, out_type = 'both'):

    #RSAS, sents, ents, break_point = RSA_adapt(test_file, vocab_file, model_files, has_header)
    RSAS, baselines, comps, sents, ents, break_point = RSA_adapt(
            test_file, vocab_file, model_files, has_header)

    header_models = []
    if not only_avg:
        for model in RSAS:
            m = model.split('/')[-1] + '_RSA_pre'
            header_models.append(m)
            m = model.split('/')[-1] + '_RSA_post'
            header_models.append(m)
            m = model.split('/')[-1] + '_RSA_diff'
            header_models.append(m)

    header_models.append('avg_RSA_pre')
    header_models.append('avg_RSA_post')
    header_models.append('avg_RSA_diff')

    header = ['baseline', 'comparison', 'LOW', 'LOW_ENT'] + header_models + ['baseline', 'comparison', 'HIGH', 'HIGH_ENT'] + header_models

    baselines_LOW = baselines[:len(baselines)//2]
    baselines_HIGH = baselines[len(baselines)//2:]

    comps_LOW = comps[:len(comps)//2]
    comps_HIGH = comps[len(comps)//2:]

    sents_LOW = sents[:len(sents)//2]
    sents_HIGH = sents[len(sents)//2:]

    ents_LOW = ents[:len(ents)//2]
    ents_HIGH = ents[len(ents)//2:]

    assert len(sents_LOW) == break_point

    all_data = []
    for x in range(len(sents_LOW)):
        data = []

        pre_lows = []
        post_lows = []
        diff_lows = []

        pre_highs = []
        post_highs = []
        diff_highs = []

        for model in RSAS:
            values = RSAS[model]

            pre_low, post_low = values[:len(values)//2][x]
            pre_high, post_high = values[len(values)//2:][x]

            pre_lows.append(pre_low)
            post_lows.append(post_low)
            diff_lows.append(post_low-pre_low)

            pre_highs.append(pre_high)
            post_highs.append(post_high)
            diff_highs.append(post_high-pre_high)

        avg_pre_low = sum(pre_lows)/len(pre_lows)
        avg_post_low = sum(post_lows)/len(post_lows)
        avg_diff_low = sum(diff_lows)/len(diff_lows)

        avg_pre_high = sum(pre_highs)/len(pre_highs)
        avg_post_high = sum(post_highs)/len(post_highs)
        avg_diff_high = sum(diff_highs)/len(diff_highs)

        if not only_avg:
            data += [baselines_LOW[x], comps_LOW[x], 
                    sents_LOW[x], ents_LOW[x]]
            data += pre_lows
            data += post_lows
            data += diff_lows
            data.append(avg_pre_low)
            data.append(avg_post_low)
            data.append(avg_diff_low)

            data += [baselines_HIGH[x], comps_HIGH[x], 
                    sents_HIGH[x], ents_HIGH[x]]
            data += pre_highs
            data += post_highs
            data += diff_highs
            data.append(avg_pre_high)
            data.append(avg_post_high)
            data.append(avg_diff_high)

        else:
            data += [baselines_LOW[x], comps_LOW[x], 
                    sents_LOW[x], ents_LOW[x]]
            data.append(avg_pre_low)
            data.append(avg_post_low)
            data.append(avg_diff_low)

            data += [baselines_HIGH[x], comps_HIGH[x], 
                    sents_HIGH[x], ents_HIGH[x]]

            data.append(avg_pre_high)
            data.append(avg_post_high)
            data.append(avg_diff_high)

        all_data.append(data)
        
    dataframe = pd.DataFrame(all_data, columns = header) 

    if out_type == 'both':
        dataframe.to_csv(out_name, index=False)
        dataframe.to_excel(out_name, index=False)

    elif out_type == 'csv':
        dataframe.to_csv(out_name, index=False)

    elif out_type == 'xlsx':
        dataframe.to_excel(out_name, index=False)

def run_adapt(test_file, vocab_file, model_files,  
        out_name, has_header = True, only_avg = False, out_type = 'both'):

    DELTAS, sents, ents, break_point = adapt(test_file, vocab_file, model_files, has_header)

    header_models = []
    if not only_avg:
        for model in DELTAS:
            model = model.split('/')[-1] + '_delta'
            header_models.append(model)

    header_models.append('avg_delta')

    header = ['LOW', 'LOW_ENT'] + header_models + ['HIGH', 'HIGH_ENT'] + header_models

    sents_LOW = sents[:len(sents)//2]
    sents_HIGH = sents[len(sents)//2:]

    ents_LOW = ents[:len(ents)//2]
    ents_HIGH = ents[len(ents)//2:]

    assert len(sents_LOW) == break_point

    all_data = []
    for x in range(len(sents_LOW)):
        data = []

        lows = []
        highs = []
        for model in DELTAS:
            values = DELTAS[model]

            delta_LOW = values[:len(values)//2][x]
            delta_HIGH = values[len(values)//2:][x]

            lows.append(delta_LOW)
            highs.append(delta_HIGH)

        avg_low = sum(lows)/len(lows)
        avg_high = sum(highs)/len(highs)

        if not only_avg:
            data += [sents_LOW[x], ents_LOW[x]]
            data += lows
            data.append(avg_low)
            data += [sents_HIGH[x], ents_HIGH[x]]
            data += highs
            data.append(avg_high)
        else:
            data += [sents_LOW[x], ents_LOW[x]]
            data.append(avg_low)
            data += [sents_HIGH[x], ents_HIGH[x]]
            data.append(avg_high)

        all_data.append(data)
        
        
    dataframe = pd.DataFrame(all_data, columns = header) 

    if out_type == 'both':
        dataframe.to_csv(out_name, index=False)
        dataframe.to_excel(out_name, index=False)

    elif out_type == 'csv':
        dataframe.to_csv(out_name, index=False)

    elif out_type == 'xlsx':
        dataframe.to_excel(out_name, index=False)

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
                try:
                    new_model = m.RNNModel('LSTM', 28439, 400, 400, 2, None, 0.2, tie_weights=True).to(device)
                    new_model.load_state_dict(model.state_dict())
                    model = new_model
                except:
                    new_model = m.RNNModel('LSTM', 50002, 400, 400, 2, None, 0.2, tie_weights=True).to(device)
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
                try:
                    new_model = m.RNNModel('LSTM', 28439, 400, 400, 2, None, 0.2, tie_weights=True).to(device)
                    new_model.load_state_dict(model.state_dict())
                    model = new_model
                except:
                    new_model = m.RNNModel('LSTM', 50002, 400, 400, 2, None, 0.2, tie_weights=True).to(device)
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

def check_unk(stim_file, vocab_file, header=False, 
        filter_file=None, verbose=False):

    ''' Given a stimuli file and model vocabulary file 
    return UNK'd stimuli.'''

    #hard code data_dir
    data_path = './'


    #Load experiments
    EXP = data.Stim(stim_file, header, filter_file, vocab_file)
    EXP.check_unks()

    return EXP

'''
stim_file = 'stimuli/Adapt.xlsx'
vocab_file = 'models/vocab' 
out_name = 'Adapt.xlsx'
model_files = glob.glob('models/*.pt')[:1]
run_RSA_adapt(stim_file, vocab_file, model_files, out_name)
'''

'''
stim_file = 'stimuli/multi_sent_another2.xlsx'
vocab_file = 'models/vocab'
model_files = glob.glob('models/*.pt')[:1]

header = True
multisent_flag = True
filter_file = None
filter_file = 'filter'
verbose = True
hasSim = True
only_avg = True

#EXP = run_norming(stim_file, vocab_file, model_files, header, multisent_flag, filter_file, verbose)
EXP = run_RSA(stim_file, vocab_file, model_files, header, multisent_flag, filter_file, verbose)
#EXP.save_csv('RSA_2_'+stim_file.split('/')[-1], model_files, only_avg, hasSim)
'''
