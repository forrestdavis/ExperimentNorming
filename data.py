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

    def return_measure(self, model_files, measure):

        out = [self.word]

        for model in model_files:
            if measure == 'sim':
                out.append(self.sims[model])
            elif measure == 'surp':
                out.append(self.surps[model])
            elif measure == 'ent':
                out.append(self.ents[model])
        return out


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

    def return_data_dict(self, model_files):
        #get averages
        if len(self.surps) == 0:
            avg_surp = -1
        else:
            avg_surp = sum(self.surps.values())/len(self.surps.values())
        if len(self.ents) == 0:
            avg_ent = -1
        else:
            avg_ent = sum(self.ents.values())/len(self.ents.values())
        if len(self.red_ents) == 0:
            avg_red_ent = -1
        else:
            avg_red_ent = sum(self.red_ents.values())/len(self.red_ents.values())
        if len(self.sims) == 0:
            avg_sim = -1
        else:
            avg_sim = sum(self.sims.values())/len(self.sims.values())

        #models X measures
        data_dict = {}
        for model in model_files:
            data_dict[model] = {}
            data_dict[model]['word'] = self.word
            data_dict[model]['surp'] = self.surps.get(model, -1)
            data_dict[model]['ent'] = self.ents.get(model, -1)
            data_dict[model]['red_ent'] = self.red_ents.get(model, -1)
            data_dict[model]['sim'] = self.sims.get(model, -1)

            data_dict[model]['surp_avg'] = avg_surp
            data_dict[model]['ent_avg'] = avg_ent
            data_dict[model]['red_ent_avg'] = avg_red_ent
            data_dict[model]['sim_avg'] = avg_sim

        return data_dict


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
            vocab_f=None):

        #Read in stimuli
        self.EXP, self.header = read_stim_file(sent_file, hasHeader)
        #Read in vocab
        print(vocab_f)
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
        self.cell_dataframe = None

    def save_plot(self, fname, model_files, hasSim=False, plot_dict=None):
        self.create_flat_df(model_files, hasSim)

        if '.' in fname[0]:
            parts = fname.split('.')
            source_fname = '.'.join(parts[:-1])+'_source.xlsx'
            plot_fname = '.'.join(parts[:-1])+'.png'
        else:
            source_fname = fname.split('.')[0]+'_source.xlsx'
            plot_fname = fname.split('.')[0]+'.png'

        
        self.dataframe.to_excel(source_fname, index=False)

        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 10))
        sns.set(font_scale=1.3)

        if plot_dict is not None:
            regions = plot_dict['regions']
        else:
            regions = {1: [2, 3, 5], 
                        2: [0, 3, 4, 5, 7], 
                        3: [0, 3, 4, 5, 7]}

        #filter dataframe
        plot_data = self.dataframe.copy()

        filter_info = None
        for region in regions:
            cur_filter = (plot_data['sentence_num'] == region) & (plot_data['word_num'].isin(regions[region]))
            if filter_info is not None:
                filter_info = filter_info | cur_filter
            else:
                filter_info = cur_filter

        plot_data = plot_data[filter_info]

        #Get percent data
        percent_model_files = list(filter(lambda x: 'percent.pt' in x,
                model_files))
        if percent_model_files:
            percent_model_files = list(map(lambda x: x.split('/')[-1].replace('.pt', ''), percent_model_files))
            percent_plot_data = plot_data[plot_data['model'].isin(percent_model_files)]

            if plot_dict and plot_dict['extra']['grand_avg']:
                percent_plot = sns.lineplot(data=percent_plot_data, x='word_id', 
                        y='sim', 
                        marker="o")
            else:
                percent_plot = sns.lineplot(data=percent_plot_data, x='word_id', 
                        y='sim', 
                        hue='model',
                        marker="o")

            x_axis_words = plot_data[(plot_data['item'] == 0) & (plot_data['model'] == model_files[0].split('/')[-1].replace('.pt', ''))]['word'].tolist()

            percent_plot.set_xticklabels(x_axis_words)
            percent_plot.set_xlabel('')
            percent_plot.set_ylabel('Similarity to Baseline')
            plt.ylim(0, 1.5)

            plt.savefig(plot_fname.replace('.png', '_percents.png'), dpi=300)
            #plt.show()

        #Get epochs data
        epoch_model_files = list(filter(lambda x: 'percent.pt' not in x,
                model_files))
        if epoch_model_files:

            #set some variables
            plt.figure(figsize=(20, 10))
            sns.set(font_scale=1.3)

            epoch_model_files = list(map(lambda x: x.split('/')[-1].replace('.pt', ''), epoch_model_files))
            epoch_plot_data = plot_data[plot_data['model'].isin(epoch_model_files)]

            if plot_dict and plot_dict['extra']['grand_avg']:
                epoch_plot = sns.lineplot(data=epoch_plot_data, x='word_id', 
                        y='sim', 
                        marker="o")
            else:
                epoch_plot = sns.lineplot(data=epoch_plot_data, x='word_id', 
                        y='sim', 
                        hue='model',
                        marker="o")

            x_axis_words = plot_data[(plot_data['item'] == 0) & (plot_data['model'] == model_files[0].split('/')[-1].replace('.pt', ''))]['word'].tolist()

            epoch_plot.set_xticklabels(x_axis_words)
            epoch_plot.set_xlabel('')
            epoch_plot.set_ylabel('Similarity to Baseline')
            plt.ylim(0, 1.5)

            plt.savefig(plot_fname.replace('.png', '_epochs.png'), dpi=300)
            #plt.show()

    def save_excel(self, fname, model_files, only_avg=False, hasSim=False):

        if '.' == fname[0]:
            parts = fname.split('.')
            fname = '.'.join(parts[:-1])+'.xlsx'
        else:
            fname = fname.split('.')[0] + '.xlsx'

        if self.dataframe is None:
            self.create_df(model_files, only_avg, hasSim)
        self.dataframe.to_excel(fname, index=False)

    def save_csv(self, fname, model_files, only_avg=False, hasSim=False):

        if '.' == fname[0]:
            parts = fname.split('.')
            fname = '.'.join(parts[:-1])+'.csv'
        else:
            fname = fname.split('.')[0] + '.csv'

        if self.dataframe is None:
            self.create_df(model_files, only_avg, hasSim)

        self.dataframe.to_csv(fname, index=False)

    def save_cell(self, fname, model_files, measure, measure_pos=-1):

        fname = fname.split('.')[0] + '.xlsx'

        self.create_cell_df(model_files, measure, measure_pos)

        self.cell_dataframe.to_excel(fname, index=False)

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

    def create_flat_out(self, model_files, measure, measure_pos=-1):

        #create header
        header = ['item']

        hasSim = False
        if measure == 'sim':
            hasSim = True

        for x in range(len(self.SENTS[0])):
            header.append('SENTS_'+str(x))
        for x in range(len(self.SENTS[0])):
            header.append('UNK_SENTS_'+str(x))
        for x in range(len(self.SENTS[0])):
            header.append('has_UNK_'+str(x))

        header.append('MODEL')
        header.append(measure)

        out_str = ','.join(header)+'\n'

        #loop over stimuli
        item = 0
        for x in range(self.TABLES[0].shape[0]):

            if x!=0 and x%2 == 0:
                item += 1

            row = [item]

            row += self.SENTS[x]
            row += self.UNK_SENTS[x]
            row += self.hasUNK[x]

            table = self.TABLES[measure_pos]

            return_value = []
            #for each word grab the measure
            for z in range(table.shape[1]):
                entry = table[x,z]
                try:
                    values = entry.return_measure(model_files, measure)
                    if values[0] == '.':
                        continue
                    return_value = values
                except:
                    break

            #skip over word
            return_value = return_value[1:]
            for i, model in enumerate(model_files):
                temp = ','.join(list(map(lambda x: str(x), row)))
                temp += ','+str(i)+','+str(return_value[i]) +'\n'

                out_str += temp
        return out_str
                
    def create_cell_df(self, model_files, measure, measure_pos=-1):

        #create header
        header = []

        for x in range(len(self.SENTS[0])):
            header.append('SENTS_'+str(x))
        for x in range(len(self.SENTS[0])):
            header.append('UNK_SENTS_'+str(x))
        for x in range(len(self.SENTS[0])):
            header.append('has_UNK_'+str(x))

        for x in range(len(model_files)):
            header.append('Network '+str(x+1)+ ' '+measure)

        #loop over stimuli
        data = []
        for x in range(self.TABLES[0].shape[0]):
            row = []

            row += self.SENTS[x]
            row += self.UNK_SENTS[x]
            row += self.hasUNK[x]

            table = self.TABLES[measure_pos]

            return_value = []
            all_values = []
            #for each word grab the measure
            for z in range(table.shape[1]):
                entry = table[x,z]
                try:
                    values = entry.return_measure(model_files, measure)
                    if values[0] == '.':
                        continue
                    #remove!
                    if values[1] == 0:
                        continue
                    return_value = values
                    all_values.append(values)
                except:
                    break

            #For running ent script
            #   Get penultimate
            '''
            if len(all_values) > 2:
                return_value = all_values[-2]
            '''
            #skip over word
            row += return_value[1:]

            data.append(row)

        self.cell_dataframe = pd.DataFrame(data, columns = header) 

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

    def create_flat_df(self, model_files, hasSim=False):
        if hasSim:
            data = {'item': [], 'baseline': [], 'unk_baseline': [], 
                    'sentence': [], 
                    'unk_sentence': [], 'hasUNK': [], 'model': [], 
                    'sentence_num': [], 'word': [], 'word_num': [], 
                    'word_id': [],
                    'surp': [], 'ent': [], 'red_ent': [], 'sim': [],
                    'surp_avg': [], 'ent_avg': [], 'red_ent_avg': [],
                    'sim_avg': []}
        else:
            data = {'item': [], 'sentence': [], 
                    'unk_sentence': [], 'hasUNK': [], 'model': [], 
                    'sentence_num': [], 'word': [], 'word_num': [], 
                    'word_id': [], 
                    'surp': [], 'ent': [], 'red_ent': [],
                    'surp_avg': [], 'ent_avg': [], 'red_ent_avg': []}

        #loop over stimuli
        for item_idx in range(self.TABLES[0].shape[0]):
            #loop over sentences in stimulus
            for sent_idx in range(len(self.SENTS[item_idx])):
                if hasSim:
                    if sent_idx == 0:
                        continue
                table = self.TABLES[sent_idx]
                #loop over each word
                for word_idx in range(table.shape[1]):
                    entry = table[item_idx, word_idx]
                    entry_data = entry.return_data_dict(model_files)

                    skipWord = 0

                    #Skip over this word, if it has no value
                    for model in model_files:
                        if entry_data[model]['surp'] == -1:
                            skipWord = 1
                        break
                    if skipWord:
                        break

                    #populate data
                    for model in model_files:

                        data['item'].append(item_idx)
                        if hasSim:
                            data['baseline'].append(self.SENTS[item_idx][0])
                            data['unk_baseline'].append(self.UNK_SENTS[item_idx][0])
                        data['sentence'].append(self.SENTS[item_idx][sent_idx])
                        data['unk_sentence'].append(self.UNK_SENTS[item_idx][sent_idx])
                        data['hasUNK'].append(self.hasUNK[item_idx][sent_idx])

                        data['model'].append(model.split('/')[-1].replace('.pt', ''))
                        data['sentence_num'].append(sent_idx)
                        data['word_num'].append(word_idx)

                        word_id = str(sent_idx).zfill(2)+str(word_idx).zfill(2)
                        data['word_id'].append(word_id)
                        
                        for key in entry_data[model]:
                            data[key].append(entry_data[model][key])

        self.dataframe = pd.DataFrame(data)

    def load_IT(self, model_name, item_idx, values, multisent_flag=False, sims=None):

        if sims:
            #break back into sentences
            if multisent_flag:
                new_sims = []
                sents = [] 
                sent_idx = 1
                length = len(self.UNK_SENTS[item_idx][sent_idx].split(' '))
                end = self.UNK_SENTS[item_idx][sent_idx].split(' ')[-1]
                sent = [] 
                new_sim = [] 
                count = 1
                for y in range(len(values[0])):
                    v = values[0][y]
                    new_sim.append(sims[0][y])
                    sent.append(v)
                    if v[0] == end and count%length == 0:
                        new_sims.append(new_sim)
                        sents.append(sent)
                        sent_idx += 1
                        if sent_idx > len(self.UNK_SENTS[item_idx])-1:
                            continue

                        end = self.UNK_SENTS[item_idx][sent_idx].split(' ')[-1]
                        length = len(self.UNK_SENTS[item_idx][sent_idx].split(' '))
                        count = 0
                        sent = []
                        new_sim = []

                    count += 1

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

        #assert everything is good
        for z in range(len(values)):
            og_sent = self.UNK_SENTS[item_idx][z]
            if not values[z]:
                continue
            t_sent = ' '.join(list(map(lambda x: x[0], values[z])))
            try:
                assert og_sent == t_sent
            except:
                assert self.SENTS[item_idx][z] == t_sent

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

