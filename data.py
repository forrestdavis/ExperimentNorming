#############################################
#This script includes a loading function and 
#class structure for formatting data and 
#norming info.
#Expects a xlsx file with no header that has 
#two columns with sentences
#Outputs xlsx with information about UNKs, 
#surprisal, and entropy
#############################################
import re
import pandas as pd
import numpy as np

DETS = {'the', 'a', 'her', 'his', 'their', 'this', 'that', 
        'its', 'another', 'other', 'some', 'all', 'many'}

class Stim:

    def __init__(self, sent_file, hasHeader=False, vocab_f='models/vocab'):

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
        
        #Extract target info
        self.load_stim()

        self.VERB_ENTROPY = {}
        self.VERB_REDUCED_ENTROPY = {}
        self.VERB_SURP = {}
        self.NOUN_SURP = {}
        self.NOUN_ENTROPY = {}
        self.dataframe = None

    def save_excel(self, fname):

        fname = fname.split('.')[0]+'.xlsx'

        if self.dataframe is None:
            self.create_df()

        self.dataframe.to_excel(fname, index=False)

    def save_csv(self, fname):
        fname = fname.split('.')[0]+'.csv'

        if self.dataframe is None:
            self.create_df()

        self.dataframe.to_csv(fname, index=False)


    #God this is so gross :(
    def create_df(self):

        if len(self.header) == 1:
            SENT1 = list(map(lambda x: x[0], self.SENTS))

            UNK_SENT1 = list(map(lambda x: x[0], self.UNK_SENTS))

            hasUNK1 = list(map(lambda x: x[0], self.hasUNK))

            MODELS1 = []
            names1 = []
            possible_names1 = ['VERB1_ENTROPY', 'VERB1_REDUCTION', 'VERB1_SURP',
                    'NOUN1_ENTROPY', 'NOUN1_SURP']

            num_models = len(self.VERB_ENTROPY.keys())
            num_obs = len(SENT1)

            #info X models X obs
            table = np.empty((len(possible_names1),num_models, num_obs), dtype=np.float64)
            labels = np.empty((len(possible_names1),num_models), dtype=object)

            y = 0
            for model in self.VERB_ENTROPY:

                #populate table
                table[0,y,:] = list(map(lambda x: x[0], self.VERB_ENTROPY[model]))
                table[1,y,:] = list(map(lambda x: x[0], self.VERB_REDUCED_ENTROPY[model]))
                table[2,y,:] = list(map(lambda x: x[0], self.VERB_SURP[model]))
                table[3,y,:] = list(map(lambda x: x[0], self.NOUN_ENTROPY[model]))
                table[4,y,:] = list(map(lambda x: x[0], self.NOUN_SURP[model]))

                x = 0
                #populate labels
                for z in range(len(possible_names1)):
                    name1 = possible_names1[z]
                    labels[x,y] = name1 + '_'+model.split('/')[-1]
                    x += 1

                y += 1

            #get avgs 
            avgs = np.empty((len(possible_names1), num_obs), dtype=np.float64)
            avg_labels = []
            for x in range(avgs.shape[0]):
                for z in range(table.shape[-1]):
                    avgs[x, z] = sum(table[x,:,z])/len(table[x,:,z])


            #Get avg_labels
            for z in range(len(possible_names1)):
                name1 = possible_names1[z]
                avg_labels.append(name1+'_AVG')

            header = ['SENT1', 'UNK_SENT1', 'hasUNK1']

            for x in range(labels.shape[0]):
                header += labels[x,:].tolist()
                header.append(avg_labels[x])

            data = []
            for x in range(len(SENT1)):
                d = [SENT1[x], UNK_SENT1[x], hasUNK1[x]]
                for y in range(table.shape[0]):
                    d += table[y,:,x].tolist()
                    d.append(avgs[y, x])
                data.append(d)

            self.dataframe = pd.DataFrame(data, columns = header) 

        if len(self.header) == 2:
            SENT1 = list(map(lambda x: x[0], self.SENTS))
            SENT2 = list(map(lambda x: x[1], self.SENTS))

            UNK_SENT1 = list(map(lambda x: x[0], self.UNK_SENTS))
            UNK_SENT2 = list(map(lambda x: x[1], self.UNK_SENTS))

            hasUNK1 = list(map(lambda x: x[0], self.hasUNK))
            hasUNK2 = list(map(lambda x: x[1], self.hasUNK))

            MODELS1 = []
            MODELS2 = []
            names1 = []
            names2 = []
            possible_names1 = ['VERB1_ENTROPY', 'VERB1_REDUCTION', 'VERB1_SURP',
                    'NOUN1_ENTROPY', 'NOUN1_SURP']
            possible_names2 = ['VERB2_ENTROPY', 'VERB2_REDUCTION', 'VERB2_SURP',
                    'NOUN2_ENTROPY', 'NOUN2_SURP']

            num_models = len(self.VERB_ENTROPY.keys())
            num_obs = len(SENT1)

            #info X models X obs
            table = np.empty((len(possible_names1)*2,num_models, num_obs), dtype=np.float64)
            labels = np.empty((len(possible_names1)*2,num_models), dtype=object)

            y = 0
            for model in self.VERB_ENTROPY:

                #populate table
                table[0,y,:] = list(map(lambda x: x[0], self.VERB_ENTROPY[model]))
                table[1,y,:] = list(map(lambda x: x[1], self.VERB_ENTROPY[model]))
                table[2,y,:] = list(map(lambda x: x[0], self.VERB_REDUCED_ENTROPY[model]))
                table[3,y,:] = list(map(lambda x: x[1], self.VERB_REDUCED_ENTROPY[model]))
                table[4,y,:] = list(map(lambda x: x[0], self.VERB_SURP[model]))
                table[5,y,:] = list(map(lambda x: x[1], self.VERB_SURP[model]))
                table[6,y,:] = list(map(lambda x: x[0], self.NOUN_ENTROPY[model]))
                table[7,y,:] = list(map(lambda x: x[1], self.NOUN_ENTROPY[model]))
                table[8,y,:] = list(map(lambda x: x[0], self.NOUN_SURP[model]))
                table[9,y,:] = list(map(lambda x: x[1], self.NOUN_SURP[model]))

                x = 0
                #populate labels
                for z in range(len(possible_names1)):
                    name1 = possible_names1[z]
                    name2 = possible_names2[z]
                    labels[x,y] = name1 + '_'+model.split('/')[-1]
                    x += 1
                    labels[x,y] = name2 + '_' + model.split('/')[-1]
                    x += 1

                y += 1

            #get avgs 
            avgs = np.empty((len(possible_names1)*2, num_obs), dtype=np.float64)
            avg_labels = []
            for x in range(avgs.shape[0]):
                for z in range(table.shape[-1]):
                    avgs[x, z] = sum(table[x,:,z])/len(table[x,:,z])


            #Get avg_labels
            for z in range(len(possible_names1)):
                name1 = possible_names1[z]
                name2 = possible_names2[z]
                avg_labels.append(name1+'_AVG')
                avg_labels.append(name2+'_AVG')

            header = ['SENT1', 'UNK_SENT1', 'hasUNK1']

            for x in range(labels.shape[0]):
                if x%2 == 0:
                    header += labels[x,:].tolist()
                    header.append(avg_labels[x])

            header += ['SENT2', 'UNK_SENT2', 'hasUNK2']
            for x in range(labels.shape[0]):
                if x%2 == 1:
                    header += labels[x,:].tolist()
                    header.append(avg_labels[x])

            data = []
            for x in range(len(SENT1)):
                d = [SENT1[x], UNK_SENT1[x], hasUNK1[x]]
                for y in range(table.shape[0]):
                    if y%2 == 0:
                        d += table[y,:,x].tolist()
                        d.append(avgs[y, x])
                d += [SENT2[x], UNK_SENT2[x], hasUNK2[x]]
                for y in range(table.shape[0]):
                    if y%2 != 0:
                        d += table[y,:,x].tolist()
                        d.append(avgs[y, x])
                data.append(d)

            self.dataframe = pd.DataFrame(data, columns = header) 


    def load_IT(self, model_name, target_idx, values, multisent_flag=False):
        
        target_words = self.TARGET_WORDS[target_idx]
        target_idxs = self.TARGET_IDX[target_idx]

        #print(target_words)

        #split values into seperate sentences
        if multisent_flag:
            values = values[0]
            values1 = []
            values2 = []
            inOne = 1
            inTwo = 0
            for v in values:
                if inOne:
                    values1.append(v)
                if inTwo:
                    values2.append(v)
                if inOne and v[0] == '.':
                    inOne = 0
                    inTwo = 1
                
            values = [values1, values2]

        #print(values)

        verb_ent = []
        verb_red = []
        verb_surp = []
        noun_entropy = []
        noun_surp = []
        for sent_idx in range(len(target_words)):
            sent_words = target_words[sent_idx]
            sent_idxs = target_idxs[sent_idx]
            IT_info = values[sent_idx]
            for x in range(len(sent_words)):
                t_word = sent_words[x]
                t_idxs = sent_idxs[x]

                IT_word = IT_info[t_idxs][0]
                IT_surp = IT_info[t_idxs][1]
                IT_entropy = IT_info[t_idxs+1][2]
                IT_red = IT_info[t_idxs][2]-IT_entropy

                assert (t_word == IT_word or IT_word == '<unk>')

                if x == 0:
                    verb_ent.append(IT_entropy)
                    verb_red.append(IT_red)
                    verb_surp.append(IT_surp)
                else:
                    noun_surp.append(IT_surp)
                    noun_entropy.append(IT_entropy)

        if model_name not in self.VERB_ENTROPY:
            self.VERB_ENTROPY[model_name] = []
        if model_name not in self.VERB_REDUCED_ENTROPY:
            self.VERB_REDUCED_ENTROPY[model_name] = []
        if model_name not in self.VERB_SURP:
            self.VERB_SURP[model_name] = []
        if model_name not in self.NOUN_SURP:
            self.NOUN_SURP[model_name] = []
        if model_name not in self.NOUN_ENTROPY:
            self.NOUN_ENTROPY[model_name] = []

        assert verb_ent != []

        self.VERB_ENTROPY[model_name].append(verb_ent)
        self.VERB_REDUCED_ENTROPY[model_name].append(verb_red)
        self.VERB_SURP[model_name].append(verb_surp)
        self.NOUN_SURP[model_name].append(noun_surp)
        self.NOUN_ENTROPY[model_name].append(noun_entropy)

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

    #Extract from EXP targets words and indices
    def load_stim(self):

        #One column
        if len(self.header) == 1:
            SENT1 = self.EXP[self.header[0]]

            #For each pair
            for x in range(len(SENT1)):
                sent1 = SENT1[x].lower().strip()
                sent1 = sent1.replace(',', ' ,').replace('.', '') + ' .'

                self.SENTS.append((sent1,))

                sent1 = sent1.split()

                target_words = []
                target_idx = []
                has_unk = [0]
                #find verb, object
                prev_word = ''
                t_words = []
                t_idx = [] 
                unk_sent1 = []
                for x in range(len(sent1)):
                    word = sent1[x]
                    if x > 1:
                        if word in DETS:
                            if prev_word not in DETS:
                                t_words.append(prev_word)
                                t_idx.append(x-1)
                        if word == '.':
                            t_words.append(prev_word)
                            t_idx.append(x-1)

                    #check for unks
                    if word not in self.model_vocab:
                        unk_sent1.append("<unk>")
                        has_unk[0] = 1
                    else:
                        unk_sent1.append(word)

                    prev_word = word

                target_words.append(t_words)
                target_idx.append(t_idx)

                self.TARGET_WORDS.append(target_words)
                self.TARGET_IDX.append(target_idx)
                self.UNK_SENTS.append((' '.join(unk_sent1),))
                self.hasUNK.append(has_unk)

        #Two columns
        if len(self.header) == 2:
            SENT1 = self.EXP[self.header[0]]
            SENT2 = self.EXP[self.header[1]]

            #For each pair
            for x in range(len(SENT1)):
                sent1 = SENT1[x].lower().strip()
                sent2 = SENT2[x].lower().strip()

                sent1 = sent1.replace(',', ' ,').replace('.', '') + ' .'
                sent2 = sent2.replace(',', ' ,').replace('.', '') + ' .'

                self.SENTS.append((sent1, sent2))

                sent1 = sent1.split()
                sent2 = sent2.split()

                target_words = []
                target_idx = []
                has_unk = [0, 0]
                #find verb, object
                prev_word = ''
                t_words = []
                t_idx = [] 
                unk_sent1 = []
                for x in range(len(sent1)):
                    word = sent1[x]
                    if x > 1:
                        if word in DETS:
                            if prev_word not in DETS:
                                t_words.append(prev_word)
                                t_idx.append(x-1)
                        if word == '.':
                            t_words.append(prev_word)
                            t_idx.append(x-1)

                    #check for unks
                    if word not in self.model_vocab:
                        unk_sent1.append("<unk>")
                        has_unk[0] = 1
                    else:
                        unk_sent1.append(word)

                    prev_word = word

                target_words.append(t_words)
                target_idx.append(t_idx)

                #find verb, object
                prev_word = ''
                t_words = []
                t_idx = [] 
                unk_sent2 = []
                for x in range(len(sent2)):
                    word = sent2[x]
                    if x > 1:
                        if word in DETS:
                            if prev_word not in DETS:
                                t_words.append(prev_word)
                                t_idx.append(x-1)
                        if word == '.':
                            t_words.append(prev_word)
                            t_idx.append(x-1)

                    #check for unks
                    if word not in self.model_vocab:
                        unk_sent2.append("<unk>")
                        has_unk[1] = 1
                    else:
                        unk_sent2.append(word)

                    prev_word = word

                target_words.append(t_words)
                target_idx.append(t_idx)

                self.TARGET_WORDS.append(target_words)
                self.TARGET_IDX.append(target_idx)
                self.UNK_SENTS.append((' '.join(unk_sent1), ' '.join(unk_sent2)))
                self.hasUNK.append(has_unk)

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
    stimf = 'stimuli/multi_sent.xlsx'
    EXP = Stim(stimf, True)

