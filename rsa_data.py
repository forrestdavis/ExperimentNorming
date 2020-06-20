#############################################
#This script includes a loading function and 
#class structure for formatting data and 
#norming info.
#Expects a xlsx file with optional header that has 
#two columns with sentences
#Outputs xlsx with information about UNKs, 
#surprisal, and entropy
#############################################
import re
import pandas as pd
import numpy as np

DETS = {'the', 'a', 'her', 'his', 'their', 'this', 'that', 
        'its', 'another', 'other', 'some', 'all', 'many'}


class Measures:

    def __init__(self):

        self.word = None
        self.surps = {}
        self.ents = {}
        self.red_ents = {}
        #RSA sim
        self.sims = {}

    def update(self, word, model, values):

        if self.word is None:
            self.word = word

        #values (surp, ent, red_ent, sim)
        if model not in self.surps:
            self.surps[model] = 0
        if model not in self.ents:
            self.ents[model] = 0
        if model not in self.red_ents:
            self.red_ents[model] = 0
        if model not in self.sims:
            self.sims[model] = 0

        assert len(values) == 4

        self.surp[model] = values[0]
        self.ents[model] = values[1]
        self.red_ents[model] = values[2]
        self.sims[model] = values[3]

    def get_avgs(self):

        surp = sum(self.surps.values())/len(self.surps.values())
        ent = sum(self.ents.values())/len(self.ents.values())
        red_ent = sum(self.red_ents.values())/len(self.red_ents.values())
        sim = sum(self.sims.values())/len(self.sims.values())

        return [surp, ent, red_ent, sim]
        
class Stim:

    def __init__(self, sent_file, hasHeader=False, filter_file=None, 
            vocab_f='models/vocab'):

        #Read in stimuli
        self.EXP, self.header = read_stim_file(sent_file, hasHeader)
        #Read in vocab
        self.model_vocab = self.load_vocab(vocab_f)
        #Pairs of sentences
        self.SENTS = []
        #Target words for pairs
        self.TARGET_WORDS = []
        #Target indices for pairs
        self.TARGET_IDX = []
        #Has Unk list
        self.hasUNK = []
        #UNK'd sentences
        self.UNK_SENTS = []
        #MAX length of each column (sent)
        self.MAX_LENS = []


        self.load_words(filter_file)
        
        self.dataframe = None

    def save_excel(self, fname, avg=False):

        fname = fname.split('.')[0]+'.xlsx'

        if self.dataframe is None:
            if self.isTemplate:
                self.create_df()
            else:
                self.create_word_df(avg)

        self.dataframe.to_excel(fname, index=False)

    def save_csv(self, fname, avg=False):
        fname = fname.split('.')[0]+'.csv'

        if self.dataframe is None:
            if self.isTemplate:
                self.create_df()
            else:
                self.create_word_df(avg)

        self.dataframe.to_csv(fname, index=False)


    def create_word_df(self, avg=False):

        #create header 
        header = ['SENT', 'UNK_SENT', 'hasUNK']

        #Add entropy
        for i in range(self.MAX_WORDS):
            head = 'word_'+str(i)
            header.append(head)
            #Add entropy
            for model in self.WORD_ENTROPY:
                head = 'word_'+str(i)+'_ENTROPY_'+model.split('/')[-1]
                if not avg:
                    header.append(head)
            head = 'word_'+str(i)+'_ENTROPY_AVG'
            header.append(head)
            #Add entropy reduction
            for model in self.WORD_REDUCED_ENTROPY:
                head = 'word_'+str(i)+'_REDUCTION_'+model.split('/')[-1]
                if not avg:
                    header.append(head)
            head = 'word_'+str(i)+'_REDUCTION_AVG'
            header.append(head)
            #Add surp
            for model in self.WORD_SURP:
                head = 'word_'+str(i)+'_SURP_'+model.split('/')[-1]
                if not avg:
                    header.append(head)
            head = 'word_'+str(i)+'_SURP_AVG'
            header.append(head)

        data = []
        for x in range(len(self.SENTS)):
            d = []
            sent = self.SENTS[x]
            unk_sent = self.UNK_SENTS[x]
            SENT = ' '.join(sent)
            UNK_SENT = ' '.join(unk_sent)
            hasUNK = sum(self.hasUNK[x])
            d += [SENT, UNK_SENT, hasUNK]

            w = UNK_SENT.split(' ')
            words = ['']*self.MAX_WORDS
            for y in range(len(words)):
                if y > len(w)-1:
                    word = ''
                else:
                    word = w[y]
                d.append(word)
                #Get entropy
                ents = []
                for model in self.WORD_ENTROPY:
                    if y > len(w)-1:
                        ent = -1
                    else:
                        ent = self.WORD_ENTROPY[model][x][y]
                    if not avg:
                        d.append(ent)
                    ents.append(ent)
                avg_ent = sum(ents)/len(ents)
                d.append(avg_ent)
                #Get reduction
                reds = []
                for model in self.WORD_REDUCED_ENTROPY:
                    if y > len(w)-1:
                        red = -1
                    else:
                        red = self.WORD_REDUCED_ENTROPY[model][x][y]
                    if not avg:
                        d.append(red)
                    reds.append(red)
                avg_red = sum(reds)/len(reds)
                d.append(avg_red)
                #Get surps
                surps = []
                for model in self.WORD_SURP:
                    if y > len(w)-1:
                        surp = -1
                    else:
                        surp = self.WORD_SURP[model][x][y]
                    if not avg:
                        d.append(surp)
                    surps.append(surp)
                avg_surp = sum(surps)/len(surps)
                d.append(avg_surp)
            data.append(d)

        self.dataframe = pd.DataFrame(data, columns = header) 

    def load_word_IT(self, model_name, target_idx, values, 
            multisent_flag=False):

        w_entropy = []
        w_surp = []
        w_red_entropy = []
        for sent in values:
            if len(sent) > self.MAX_WORDS:
                self.MAX_WORDS = len(sent)
            for y in range(len(sent)):
                try:
                    ent = sent[y+1][-1]
                except:
                    ent = 0
                w_entropy.append(ent)
                w_surp.append(sent[y][1])
                w_red_entropy.append(max(sent[y][-1]-ent, 0))

        if model_name not in self.WORD_ENTROPY:
            self.WORD_ENTROPY[model_name] = []
        if model_name not in self.WORD_REDUCED_ENTROPY:
            self.WORD_REDUCED_ENTROPY[model_name] = []
        if model_name not in self.WORD_SURP:
            self.WORD_SURP[model_name] = []

        assert w_entropy != []

        self.WORD_ENTROPY[model_name].append(w_entropy)
        self.WORD_REDUCED_ENTROPY[model_name].append(w_red_entropy)
        self.WORD_SURP[model_name].append(w_surp)


    def load_IT(self, model_name, item_idx, values, multisent_flag=False):

        #break back into sentences
        if multisent_flag:
            sents = [] 
            sent_idx = 0
            end = self.UNK_SENTS[item_idx][sent_idx].split(' ')[-1]
            print(end)
            sent = [] 
            for v in values[0]:
                sent.append(v)
                if v[0] == end:
                    sents.append(sent)
                    sent_idx += 1
                    if sent_idx > len(self.UNK_SENTS[item_idx])-1:
                        continue

                    end = self.UNK_SENTS[item_idx][sent_idx].split(' ')[-1]
                    sent = []

            values = sents

        print(values)
        print()
        print()
        for x in range(len(self.TABLES)):
            table = self.TABLES[x]
            target_words = self.TARGET_WORDS[item_idx][x]
            target_idxs = self.TARGET_IDX[item_idx][x]
            IT = values[x]

            print(target_words)
            print(target_idxs)
            print(IT)
            print()


    def load_vocab(self, vocab):

        #load vocab
        info = set([])
        vocab = open(vocab, 'r')
        for line in vocab:
            line = line.strip()
            info.add(line)
        vocab.close()
        vocab = info
        return vocab

    def load_filter(self, f_filter):

        info = set([])
        with open(f_filter, 'r') as f:
            for line in f:
                line = line.strip().lower()
                info.add(line)

        return info

    def load_words(self, filter_file):

        #filter out possibly
        if filter_file:
            exclude = self.load_filter(filter_file)
        else:
            exclude = set([])

        #step through ITEMS
        num_stim = len(self.EXP[self.header[0]])
        max_lens = [0]*len(self.header)
        for x in range(num_stim):

            SENTS = []
            UNK_SENTS = []
            hasUNK = []
            TARGET_WORDS = []
            TARGET_IDX = []

            #step through sentences
            col_idx = 0
            for col in self.header:
                
                sent = self.EXP[col][x].lower().strip()
                sent = sent.replace(',', ' ,').replace('.', ' .')

                SENTS.append(sent)

                sent = sent.split(' ')

                has_unk = 0
                unk_sent = []
                target_words = []
                target_idxs = []

                for y in range(len(sent)):
                    word = sent[y]

                    if word not in self.model_vocab:
                        has_unk = 1
                        unk_sent.append("<unk>")
                    else:
                        unk_sent.append(word)

                    if word not in exclude:
                        target_words.append(word)
                        target_idxs.append(y)

                hasUNK.append(has_unk)
                UNK_SENTS.append(' '.join(unk_sent))
                TARGET_WORDS.append(target_words)
                TARGET_IDX.append(target_idxs)

                if len(target_words) > max_lens[col_idx]:
                    max_lens[col_idx] = len(target_words)
                col_idx += 1

            self.SENTS.append(SENTS)
            self.hasUNK.append(hasUNK)
            self.UNK_SENTS.append(UNK_SENTS)
            self.TARGET_WORDS.append(TARGET_WORDS)
            self.TARGET_IDX.append(TARGET_IDX)

        self.MAX_LENS = max_lens

        TABLES = []
        for z in range(len(max_lens)):
            num_words = max_lens[z]
            TABLES.append(np.full((num_stim, num_words), Measures))

        self.TABLES = TABLES
        return 

#Read in stimuli files as pandas dataframe
def read_stim_file(stim_file, hasHeader=False):

    if hasHeader:
        EXP = pd.read_excel(stim_file)
    else:
        EXP = pd.read_excel(stim_file, header=None)

    header = EXP.columns.values

    return EXP, header

if __name__ == "__main__":

    #stimf = 'stimuli/The_boy_will_bounce_the_ball.xlsx'
    stimf = 'stimuli/RSA_Analysis.xlsx'
    filter_file = None
    EXP = Stim(stimf, True, filter_file)

