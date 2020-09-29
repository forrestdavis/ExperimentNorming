#############################################
#This script includes a loading function and 
#class structure for formatting data and 
#norming info.
#Expects a xlsx file with optional header that 
#has sentences in seperate columns
#Outputs xlsx with information about UNKs, 
#surprisal, entropy, entropy reduction
#Additional format is similarity, with 
#target in first column and sentences to 
#get word-by-word similarity for
#############################################
import re
import pandas as pd
import numpy as np
import sys

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
        else:
            assert self.word == word

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

        self.surps[model] = values[0]
        self.ents[model] = values[1]
        self.red_ents[model] = values[2]
        self.sims[model] = values[3]

    def get_avgs(self):

        '''
        if type(self.surps.values) != list:
            num_surps = 1
        else:
            num_surps = len(self.surps.values())

        if type(self.ents.values) != list:
            num_ents = 1
        else:
            num_ents = len(self.ents.values())

        if type(self.red_ents.values) != list:
            num_red_ents = 1
        else:
            num_red_ents = len(self.red_ents.values())

        if type(self.sims.values) != list:
            num_sims = 1
        else:
            num_sims = len(self.sims.values())
        '''

        surp = sum(self.surps.values())/len(self.surps.values())
        ent = sum(self.ents.values())/len(self.ents.values())
        red_ent = sum(self.red_ents.values())/len(self.red_ents.values())
        sim = sum(self.sims.values())/len(self.sims.values())

        return [surp, ent, red_ent, sim]

    def return_models(self):
        return list(self.surps.keys())
        
    def print(self):

        out = [self.word]

        surp, ent, red_ent, sim = self.get_avgs()

        out.append(ent)
        out.append(red_ent)
        out.append(surp)
        if sim:
            out.append(sim)

        for model in self.surps:
            out.append(self.ents[model])
            out.append(self.red_ents[model])
            out.append(self.surps[model])
            if sim:
                out.append(self.sims[model])

        out = list(map(lambda x: str(x), out))
        out_str = ','.join(out)
        print(out_str)

    def return_data(self, model_files, only_avg=False): 
        out = [self.word]

        surp, ent, red_ent, sim = self.get_avgs()

        out.append(ent)
        out.append(red_ent)
        out.append(surp)
        if sim:
            out.append(sim)

        if only_avg:
            return out

        for model in model_files:
            out.append(self.ents[model])
            out.append(self.red_ents[model])
            out.append(self.surps[model])
            if sim:
                out.append(self.sims[model])
        return out

    def return_blank(self, model_files, only_avg=False, hasSim=False):

        out = ['NULL']

        surp, ent, red_ent, sim = [-1, -1, -1, -1]

        out.append(ent)
        out.append(red_ent)
        out.append(surp)
        if hasSim:
            out.append(sim)

        if only_avg:
            return out

        for model in model_files:
            out.append(-1)
            out.append(-1)
            out.append(-1)
            if hasSim:
                out.append(-1)
        return out

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

        #HOLDS IT
        self.TABLES = None

        self.load_words(filter_file)

        self.dataframe = None

    def save_excel(self, fname, model_files, only_avg=False, hasSim=False):

        fname = fname.split('.')[0]+'.xlsx'

        if self.dataframe is None:
            self.create_df(model_files, only_avg, hasSim)
        self.dataframe.to_excel(fname, index=False)

    def save_csv(self, fname, model_files, only_avg=False, hasSim=False):
        fname = fname.split('.')[0]+'.csv'

        if self.dataframe is None:
            self.create_df(model_files, only_avg, hasSim)

        self.dataframe.to_csv(fname, index=False)


    def check_unks(self):

        header = []
        for x in range(len(self.SENTS[0])):
            header.append('SENTS_'+str(x))
        for x in range(len(self.SENTS[0])):
            header.append('UNK_SENTS_'+str(x))
        for x in range(len(self.SENTS[0])):
            header.append('has_UNK_'+str(x))

        #loop over stimuli
        data = []
        for x in range(self.TABLES[0].shape[0]):
            row = []

            row += self.SENTS[x]
            row += self.UNK_SENTS[x]
            row += self.hasUNK[x]

            data.append(row)

        self.dataframe = pd.DataFrame(data, columns = header) 

    def create_df(self, model_files, only_avg=False, hasSim=False):

        #create header 
        header = []

        for x in range(len(self.SENTS[0])):
            header.append('SENTS_'+str(x))
        for x in range(len(self.SENTS[0])):
            header.append('UNK_SENTS_'+str(x))
        for x in range(len(self.SENTS[0])):
            header.append('has_UNK_'+str(x))

        for x in range(len(self.SENTS[0])):
            if hasSim:
                if x == 0:
                    continue
            sent_loc = 'sent_'+str(x)+'_'
            for z in range(self.TABLES[x].shape[1]):
                w = sent_loc+'word_'+str(z)
                header.append(w)
                header.append(w+'_ent_avg')
                header.append(w+'_red_avg')
                header.append(w+'_surp_avg')
                if hasSim:
                    header.append(w+'_sim_avg')

                if only_avg:
                    continue
                
                for model in model_files:
                    model = model.split('/')[-1]
                    header.append(w+'_ent_'+model)
                    header.append(w+'_red_'+model)
                    header.append(w+'_surp_'+model)
                    if hasSim:
                        header.append(w+'_sim_'+model)


        #loop over stimuli
        data = []
        for x in range(self.TABLES[0].shape[0]):
            row = []

            row += self.SENTS[x]
            row += self.UNK_SENTS[x]
            row += self.hasUNK[x]

            #loop over sents in stimulus
            for y in range(len(self.SENTS[x])):
                if hasSim:
                    if y == 0:
                        continue
                table = self.TABLES[y]

                #for each word
                for z in range(table.shape[1]):
                    entry = table[x, z]
                    #entry.print()
                    try:
                        row += entry.return_data(model_files, only_avg)
                    except:
                        row += entry.return_blank(model_files, only_avg, hasSim)

            data.append(row)

        self.dataframe = pd.DataFrame(data, columns = header) 

    def load_IT(self, model_name, item_idx, values, multisent_flag=False, sims=None):

        if sims:
            #break back into sentences
            if multisent_flag:
                new_sims = []
                sents = [] 
                sent_idx = 1
                end = self.UNK_SENTS[item_idx][sent_idx].split(' ')[-1]
                sent = [] 
                new_sim = [] 
                for y in range(len(values[0])):
                    v = values[0][y]
                    new_sim.append(sims[0][y])
                    sent.append(v)
                    if v[0] == end:
                        new_sims.append(new_sim)
                        sents.append(sent)
                        sent_idx += 1
                        if sent_idx > len(self.UNK_SENTS[item_idx])-1:
                            continue

                        end = self.UNK_SENTS[item_idx][sent_idx].split(' ')[-1]
                        sent = []
                        new_sim = []

                values = sents
                sims = new_sims

            #add dummy
            values.insert(0, [])
            sims.insert(0, [])

        else:
            #break back into sentences
            if multisent_flag:
                sents = [] 
                sent_idx = 0
                end = self.UNK_SENTS[item_idx][sent_idx].split(' ')[-1]
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

        #add to tables
        for x in range(len(self.TABLES)):
            if sims:
                if x == 0:
                    continue
            table = self.TABLES[x]
            target_words = self.TARGET_WORDS[item_idx][x]
            target_idxs = self.TARGET_IDX[item_idx][x]
            IT = values[x]

            for y in range(len(target_idxs)):
                t_word = target_words[y]
                IT_word = IT[target_idxs[y]]
                assert IT_word[0] == t_word or IT_word[0] == '<unk>'

                surp = IT_word[1]
                try:
                    ent = IT[target_idxs[y]+1][-1]
                except:
                    ent = 0
                red_ent = max(IT_word[-1]-ent, 0)
                if sims:
                    measures = (surp, ent, red_ent, sims[x][target_idxs[y]][-1])
                else:
                    measures = (surp, ent, red_ent, 0)

                table[item_idx][y].update(t_word, model_name, measures)
                #table[item_idx][y].print()


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
                
                try:
                    sent = self.EXP[col][x].lower().strip()
                except:
                    sys.stderr.write('Columns with numbers will turned into strings and likely UNKd.' + 
                    ' You should likely delete the offending column.\n')
                    sent = str(self.EXP[col][x]).lower().strip()
                sent = sent.replace(',', ' ,').replace('.', ' .')
                #if '.' not in sent:
                #    sent += ' .'

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
            table = np.empty((num_stim, num_words), dtype=object)
            for i in range(num_stim):
                for j in range(num_words):
                    table[i, j] = Measures()
            TABLES.append(table)

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

