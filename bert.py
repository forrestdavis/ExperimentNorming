import math
import sys
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import data
import numpy as np
from transformers import BertTokenizer, BertModel
from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel, TransfoXLModel, GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModel, AutoTokenizer, RobertaForMaskedLM
import dill
#For ELMo
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.data import Token, Vocabulary
from allennlp.data.fields import ListField, TextField
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder


#device = torch.device("cuda:1")
VOCAB_FILE = 'large_models/wikitext103_vocab'

##ELMo##
def run_ELMo_RSA(stim_file, header=False, filter_file=None):

    EXP = data.Stim(stim_file, header, filter_file, VOCAB_FILE)

    #Get tokenizer
    tokenizer = WhitespaceTokenizer()

    #Load model
    ##ELMo OG
    elmo_weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
    elmo_options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json'

    #ELMo Small
    #elmo_weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
    #elmo_options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'

    #ELMo Medium
    #elmo_weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5'
    #elmo_options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json'
    
    #ELMo OG (5.5B)
    #elmo_weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
    #elmo_options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'

    elmo_embedding = ElmoTokenEmbedder(options_file=elmo_options_file, 
                                        weight_file=elmo_weight_file, dropout=0.0)
    embedder = BasicTextFieldEmbedder(token_embedders={'elmo_tokens': elmo_embedding})


    for x in range(len(EXP.SENTS)):
        sentences = list(EXP.SENTS[x])
        target = sentences[0]
        sentence = sentences[1]

        #GET BASELINE
        token_indexer = ELMoTokenCharactersIndexer()
        vocab = Vocabulary()

        target_tokens = tokenizer.tokenize(target)
        target_text_field = TextField(target_tokens, {'elmo_tokens': token_indexer})
        target_text_field.index(vocab)
        target_token_tensor = target_text_field.as_tensor(target_text_field.get_padding_lengths())
        target_tensor_dict = target_text_field.batch_tensors([target_token_tensor])

        target_embedding = embedder(target_tensor_dict)[0]
        baseline = target_embedding[-1].data.cpu().squeeze()

        #GET SIMS
        sims = get_ELMo_sims(sentence, baseline, tokenizer, embedder)
        values = get_dummy_values(sentence)

        EXP.load_IT('elmo', x, values, False, sims)

    return EXP

def get_ELMo_sims(sent, baseline, tokenizer, embedder):

    token_indexer = ELMoTokenCharactersIndexer()
    vocab = Vocabulary()

    tokens = tokenizer.tokenize(sent)
    text_field = TextField(tokens, {'elmo_tokens': token_indexer})
    text_field.index(vocab)
    token_tensor = text_field.as_tensor(text_field.get_padding_lengths())
    tensor_dict = text_field.batch_tensors([token_tensor])

    embedding = embedder(tensor_dict)[0]

    sent_words = sent.split(' ')

    SIMS = []
    idx = 0

    sims = []
    
    #offset if word is broken down
    for y in range(len(sent_words)):

        sent_word = sent_words[y]

        assert sent_word == tokens[y].text

        h = embedding[y].data.cpu().squeeze()
        assert len(h) == len(baseline)

        sim = np.corrcoef(baseline, h)[0, 1]

        sims.append((sent_word, sim))

    SIMS.append(sims)

    return SIMS

##TF XL##
def run_TFXL_RSA(stim_file, layer, header=False, filter_file=None):

    EXP = data.Stim(stim_file, header, filter_file, VOCAB_FILE)

    #Get tokenizer
    tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')

    #Load model
    model = TransfoXLModel.from_pretrained('transfo-xl-wt103', 
                                        output_hidden_states = True)#, force_download=True)
    #turn off learning
    model.zero_grad()

    for x in range(len(EXP.SENTS)):
        sentences = list(EXP.SENTS[x])
        target = sentences[0]
        sentence = sentences[1]

        #GET BASELINE
        target_encoded = tokenizer.encode(target, add_special_tokens=True)
        target_input_ids = torch.tensor(target_encoded).unsqueeze(0)

        #Get model outputs
        output = model(target_input_ids)
        predictions, mems, hidden_states = output

        hidden_states = hidden_states[1:]

        baseline = hidden_states[layer][0][-1].data.cpu().squeeze()

        #GET SIMs
        sims = get_TFXL_sims(sentence, layer, baseline, tokenizer, model)
        values = get_dummy_values(sentence)
        
        EXP.load_IT('tfxl', x, values, False, sims)

    return EXP


def get_TFXL_sims(sent, layer, baseline, tokenizer, model):

    model.zero_grad()

    encoded = tokenizer.encode(sent)
    input_ids = torch.tensor(encoded).unsqueeze(0)

    output = model(input_ids)
    predictions, mems, hidden_states = output

    hidden_states = hidden_states[1:][layer][0]

    sent_words = sent.split(' ')

    SIMS = []
    idx = 0

    sims = []
    #offset if word is broken down
    for y in range(len(sent_words)):

        sent_word = sent_words[y]
        h_idx = encoded[idx]
        input_word = tokenizer.decode(torch.tensor([h_idx])).strip()
        try:
            assert input_word == sent_word
        except: 
            assert input_word == '<unk>'

        h = hidden_states[idx].unsqueeze(0).data.cpu().squeeze()

        assert len(h) == len(baseline)

        sim = np.corrcoef(baseline, h)[0, 1]

        sims.append((sent_word, sim))

        idx += 1

    SIMS.append(sims)
    return SIMS

##GPT-2 XL##
def run_GPT_RSA(stim_file, layer, header=False, filter_file=None):

    EXP = data.Stim(stim_file, header, filter_file, VOCAB_FILE)

    #Get tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')

    #Load model
    model = GPT2LMHeadModel.from_pretrained('gpt2-xl', 
                                        output_hidden_states = True)#, force_download=True)
    #turn off learning
    model.zero_grad()

    for x in range(len(EXP.SENTS)):
        sentences = list(EXP.SENTS[x])
        target = sentences[0]
        sentence = sentences[1]

        #GET BASELINE
        target_encoded = tokenizer.encode(target, add_special_tokens=True, 
                add_prefix_space=True)
        target_input_ids = torch.tensor(target_encoded).unsqueeze(0)

        #Get model outputs
        output = model(target_input_ids)
        predictions, mems, hidden_states = output

        hidden_states = hidden_states[1:]

        baseline = hidden_states[layer][0][-1].data.cpu().squeeze()

        #GET SIMs
        sims = get_GPT_sims(sentence, layer, baseline, tokenizer, model)
        values = get_dummy_values(sentence)

        EXP.load_IT('gpt2', x, values, False, sims)

    return EXP


def get_GPT_sims(sent, layer, baseline, tokenizer, model):

    model.zero_grad()

    encoded = tokenizer.encode(sent)
    input_ids = torch.tensor(encoded).unsqueeze(0)

    output = model(input_ids)
    predictions, mems, hidden_states = output

    hidden_states = hidden_states[1:][layer][0]

    sent_words = sent.split(' ')

    SIMS = []
    idx = 0

    sims = []
    #offset if word is broken down
    for y in range(len(sent_words)):

        sent_word = sent_words[y]
        h_idx = encoded[idx]
        input_word = tokenizer.decode(torch.tensor([h_idx])).strip()
        if input_word != sent_word:
            while not(sent_word.index(input_word)+len(input_word) == len(sent_word)):
                idx += 1
                h_idx = encoded[idx]
                input_word = tokenizer.decode(torch.tensor([h_idx])).strip()

        h = hidden_states[idx].unsqueeze(0).data.cpu().squeeze()

        assert len(h) == len(baseline)

        sim = np.corrcoef(baseline, h)[0, 1]

        sims.append((sent_word, sim))

        idx += 1

    SIMS.append(sims)
    return SIMS

##BERT##
def run_BERT_RSA(stim_file, layer, header=False, filter_file=None):

    EXP = data.Stim(stim_file, header, filter_file, VOCAB_FILE)

    #Load BERT uncased 
    pretrained_weights = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    model = BertModel.from_pretrained(pretrained_weights, 
                                        output_hidden_states=True)

    #tokenizer = AutoTokenizer.from_pretrained("nyu-mll/roberta-base-100M-3")
    #tokenizer = AutoTokenizer.from_pretrained("nyu-mll/roberta-base-1B-3")
    #model = RobertaForMaskedLM.from_pretrained("nyu-mll/roberta-base-1B-3", output_hidden_states=True)

    model.eval()
    model.zero_grad()

    for x in range(len(EXP.SENTS)):
        sentences = list(EXP.SENTS[x])

        target = sentences[0]
        sentences = sentences[1:]

        #GET BASELINE
        target_encoded = tokenizer.encode(target)
        target_ids = torch.tensor(target_encoded).unsqueeze(0)

        hidden_states = model(target_ids)[-1]
        embed, hidden_states = hidden_states[:1], hidden_states[1:]

        hidden_states = hidden_states[layer][0]
        
        baseline_word = tokenizer.decode(torch.tensor([target_encoded[-2]])).strip()

        baseline = hidden_states[-2].data.cpu().squeeze()

        sims = get_BERT_sims(sentences[0], layer, baseline, tokenizer, model)
        values = get_dummy_values(sentences[0])

        EXP.load_IT('bert-uncased', x, values, False, sims)

    return EXP

def get_BERT_sims(sent, layer, baseline, tokenizer, model):

    model.zero_grad()

    encoded = tokenizer.encode(sent)
    input_ids = torch.tensor(encoded).unsqueeze(0)

    hidden_states = model(input_ids)[-1]
    embed, hidden_states = hidden_states[:1], hidden_states[1:]

    hidden_states = hidden_states[layer][0]

    #skip over [CLS] [SEP]
    hidden_states = hidden_states[1:-1]

    encoded = encoded[1:-1]

    sent_words = sent.split(' ')

    SIMS = []
    idx = 0

    sims = []
    #offset if word is broken down
    for y in range(len(sent_words)):

        sent_word = sent_words[y]
        h_idx = encoded[idx]
        input_word = tokenizer.decode(torch.tensor([h_idx])).strip()
        #replace tokenizer flag
        input_word = input_word.replace("##", '')

        if input_word != sent_word:
            while not(sent_word.index(input_word)+len(input_word) == len(sent_word)):
                idx += 1
                h_idx = encoded[idx]
                input_word = tokenizer.decode(torch.tensor([h_idx])).strip()
                #replace tokenizer flag
                input_word = input_word.replace("##", '')

        h = hidden_states[idx].unsqueeze(0).data.cpu().squeeze()

        assert len(h) == len(baseline)

        sim = np.corrcoef(baseline, h)[0, 1]

        sims.append((sent_word, sim))

        idx += 1

    SIMS.append(sims)

    return SIMS

def get_dummy_values(sent):

    values = []

    metrics = []
    for word in sent.split(' '):
        metrics.append((word, 99999, 99999))
    values.append(metrics)


    return values

if __name__ == "__main__":

    stim_file = 'stimuli/large_degree.xlsx'
    header = True
    #run_ELMo_RSA(stim_file, header)
    layer = 1

    EXP = run_TFXL_RSA(stim_file, layer, header)
    '''
    output_file = 'fu_gpt'
    model_files = ['gpt2']
    #EXP.save_cell(output_file, model_files, 'sim')
    dill.dump(EXP, file = open(output_file+'.pkl', 'wb'))
    '''
