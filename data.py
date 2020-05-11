#############################################
#This script includes a loading function and 
#class structure for formatting data and 
#norming info.
#Expects a xlsx file with no header that has 
#two columns with sentences
#Outputs xlsx with information about UNKs, 
#surprisal, entropy, and frequency
#############################################
import re
import pandas as pd

class Stim:

    def __init__(self, sent_file):

        self.EXP = read_stim_file(sent_file)


    def load_RSA(self, sim_scores, modelf):

        if modelf not in self.RSA:
            self.RSA[modelf] = 0 
        if modelf not in self.RSA1:
            self.RSA1[modelf] = 0 
        self.RSA[modelf] = sim_scores[0]
        self.RSA1[modelf] = sim_scores[1]

    #Break into target-context and seperate punctuation
    def load_stim(self, sent_targs, dataType):

        #Formatting for passive experiment
        if dataType.lower() == 'passive':
            self.MinTarget = ''
            self.MinVerb = ''

            self.SubTarget = ''
            self.SubVerb = ''

            for x in range(len(sent_targs)):

                sent = sent_targs[x]

                verb = []

                #Get verb (sluggish but ~shrug~) gets particles
                prev_word = ''
                isVerb = 0

                for word in sent.split(' '):
                    if prev_word == 'be':
                        verb.append(word)
                        isVerb = 1
                    elif isVerb:
                        if word == 'the' or word == 'his' or word == 'her' or word == "its":
                            break
                        verb.append(word)
                    prev_word = word

                if x == 0:
                    self.MinTarget = sent
                    self.MinVerb = ' '.join(verb)
                else:
                    self.SubTarget = sent
                    self.SubVerb = ' '.join(verb)

            self.vocab.update(self.MinTarget.split(' '))
            self.vocab.update(self.SubTarget.split(' '))

        #format for original experiment
        if dataType.lower() == 'full':
            self.MinTarget = ''
            self.MinFuture = ''
            self.MinVerb = ''

            self.SubTarget = ''
            self.SubFuture = ''
            self.SubVerb = ''

            for x in range(len(sent_targs)):
                sent_targ = sent_targs[x] 
                sent_targ = sent_targ.strip().split('.')
                sent = sent_targ[0].lower() + ' .'
                sent2 = sent_targ[1].lower() + ' .'
                verb = []

                #Get verb (sluggish but ~shrug~) gets particles
                prev_word = ''
                isVerb = 0
                for word in sent.split(' '):
                    if prev_word == 'will':
                        verb.append(word)
                        isVerb = 1
                    elif isVerb:
                        if word == 'the' or word == 'his' or word == 'her' or word == "its":
                            break
                        verb.append(word)
                    prev_word = word
                
                if x == 0:
                    self.MinTarget = sent.strip()
                    self.MinFuture = sent2.strip()
                    self.MinVerb = ' '.join(verb)
                else:
                    self.SubTarget = sent.strip()
                    self.SubFuture = sent2.strip()
                    self.SubVerb = ' '.join(verb)
            #update self.vocab
            self.vocab.update(self.MinTarget.split(' '))
            self.vocab.update(self.SubTarget.split(' '))

    def get_csv_header(self, flatten=False):

        if flatten:
            header = 'ITEM,MODEL,COND,SENTENCE,RSA,RATED CHANGE,RATED IMAGEABILITY,BOLD\n'
        else:
            header = 'ITEM,MODEL,COND,SENTENCE,RSA,RATED CHANGE,RATED IMAGEABILITY,BOLD,COND,SENTENCE,RSA,CHANGE,RATED IMAGEABILITY,BOLD\n'

        return header

    def get_csv_entry(self, flatten=False, avg=False):
        out = ''
        if avg:
            RSA = str(sum(self.RSA.values())/len(self.RSA.values()))
            RSA1 = str(sum(self.RSA1.values())/len(self.RSA1.values()))
            BOLD = str(self.BOLD)
            BOLD1 = str(self.BOLD1)
            Rating = str(self.Rating)
            Rating1 = str(self.Rating)
            Image = str(self.Image)
            Image1 = str(self.Image1)
            Item = str(self.Item)
            model = 'all'

            first_half = ','.join([Item, model, self.Cond, 
                self.MinTarget,RSA,Rating,Image,BOLD])
            if flatten:
                second_half = '\n'+','.join([Item, model, self.Cond1, 
                    self.SubTarget, RSA1, Rating1, Image1, BOLD1])
            else:
                second_half = ","+','.join([self.Cond1, 
                    self.SubTarget, RSA1,Rating1, Image1, BOLD1])
            out += first_half+second_half + '\n'
        else:
            for model in self.RSA:
                RSA = str(self.RSA[model])
                RSA1 = str(self.RSA1[model])
                BOLD = str(self.BOLD)
                BOLD1 = str(self.BOLD1)
                Rating = str(self.Rating)
                Rating1 = str(self.Rating1)
                Image = str(self.Image)
                Image1 = str(self.Image1)
                Item = str(self.Item)
                model = model.split('/')[-1]

                first_half = ','.join([Item, model, self.Cond, 
                    self.MinTarget,RSA,Rating,Image,BOLD])
                if flatten:
                    second_half = '\n'+','.join([Item, model, self.Cond1, 
                        self.SubTarget, RSA1, Rating1,Image1, BOLD1])
                else:
                    second_half = ","+','.join([self.Cond1, 
                        self.SubTarget, RSA1,Rating1, Image1,BOLD1])
                out += first_half+second_half + '\n'
        return out

    def get_multi_test_sents(self):

        #Awful I know, the point is to strip of the and then and the pronoun
        MinFuture = ' '.join(self.MinFuture.split(' ')[3:])
        SubFuture = ' '.join(self.SubFuture.split(' ')[3:])

        #Get the subject of the target sentence
        MinSubject = ' '.join(self.MinTarget.split(' ')[:2])
        SubSubject = ' '.join(self.SubTarget.split(' ')[:2])

        #Create sentence
        Min = MinSubject + ' '+MinFuture + ' and then , '+self.MinTarget

        Sub = SubSubject + ' '+SubFuture + ' and then , '+self.SubTarget

        return [Min, Sub]

    def get_test_sents(self):
        MinNoun = []
        inNoun = 0
        count = 0
        for word in self.MinTarget.split(' '):
            count += 1
            if count < 3:
                continue
            if word == '.':
                continue
            if word == 'the' or word == 'his' or word == 'her' or word == "its":
                inNoun = 1
                continue
            if inNoun:
                MinNoun.append(word)

        SubNoun = []
        count = 0
        inNoun = 0
        for word in self.SubTarget.split(' '):
            count += 1
            if count < 3:
                continue
            if word == '.':
                continue
            if word == 'the' or word == 'his' or word == 'her' or word == 'its':
                inNoun = 1
                continue
            if inNoun:
                SubNoun.append(word)

        MinNoun = ' '.join(MinNoun)
        SubNoun = ' '.join(SubNoun)
        if MinNoun == SubNoun:
            #comparer = "there is a "+MinNoun +' .'
            comparer = MinNoun
            self.MinNoun = MinNoun
            self.SubNoun = MinNoun
            return [self.MinTarget, self.SubTarget, comparer]
        #MinComparer = "there is a "+MinNoun +' .'
        #SubComparer = "there is a "+SubNoun + ' .'
        MinComparer = MinNoun
        SubComparer = SubNoun
        self.MinNoun = MinNoun
        self.SubNoun = SubNoun
        return [self.MinTarget, self.SubTarget, MinComparer, SubComparer]


    def check_for_unks(self, vocab):

        #load vocab
        info = set([])
        vocab = open(vocab, 'r')
        for line in vocab:
            line = line.strip()
            info.add(line)
        vocab.close()
        vocab = info

        #Check if words in stimuli missing from vocab
        if not self.vocab.issubset(vocab):
            missing_words = self.vocab.difference(vocab)
            return missing_words
        return set([])

    def filter_noun_freq(self, vocab):
        #load vocab
        info = set([])
        vocab = open(vocab, 'r')
        for line in vocab:
            line = line.strip()
            info.add(line)
        vocab.close()
        vocab = info

        #get nouns
        self.get_test_sents()

        #Check if verb is in frequent list
        if self.MinNoun not in vocab or self.SubNoun not in vocab:
            self.hasREMOVE = 1

        #Check if words in stimuli missing from vocab
        return

    def filter_verb_freq(self, vocab):
        #load vocab
        info = set([])
        vocab = open(vocab, 'r')
        for line in vocab:
            line = line.strip()
            info.add(line)
        vocab.close()
        vocab = info

        #Check if verb is in frequent list
        if self.MinVerb not in vocab or self.SubVerb not in vocab:
            self.hasREMOVE = 1

        #Check if words in stimuli missing from vocab
        return

    def filter_unks(self, vocab):
        #load vocab
        info = set([])
        vocab = open(vocab, 'r')
        for line in vocab:
            line = line.strip()
            info.add(line)
        vocab.close()
        vocab = info

        #Check if words in stimuli missing from vocab
        if not self.vocab.issubset(vocab):
            self.hasREMOVE = 1
        return

    def replace_unks(self, SWAPS):

        #check if words in stimuli are in swaps
        unks = self.vocab.intersection(set(SWAPS.keys()))
        if 'rubik' in self.MinTarget:
            self.MinTarget = self.MinTarget.replace('rubik’s', '')
            self.MinFuture = self.MinFuture.replace('rubik’s', '')
            self.SubTarget = self.SubTarget.replace('rubik’s', '')
            self.SubFuture = self.SubFuture.replace('rubik’s', '')

            self.hasREMOVE = 1
        if unks:
            for unk in unks:
                if SWAPS[unk] == 'REMOVE':

                    self.hasREMOVE = 1

                self.MinTarget = self.MinTarget.replace(unk, SWAPS[unk])
                self.MinFuture = self.MinFuture.replace(unk, SWAPS[unk])
                self.SubTarget = self.SubTarget.replace(unk, SWAPS[unk])
                self.SubFuture = self.SubFuture.replace(unk, SWAPS[unk])
                #Uncomment to remove all stimuli that 
                #have any word not in vocab
                self.hasREMOVE = 1


        #Fix spacing issue :(
        self.MinTarget = self.MinTarget.replace('  ', ' ')
        self.MinFuture = self.MinFuture.replace('  ', ' ')
        self.SubTarget = self.SubTarget.replace('  ', ' ')
        self.SubFuture = self.SubFuture.replace('  ', ' ')
        return

#Read in stimuli files as pandas dataframe
def read_stim_file(stim_file):

    EXP = pd.read_excel(stim_file, header=None)

    return EXP 

if __name__ == "__main__":

    stimf = 'stimuli/The_boy_will_bounce_the_ball.xlsx'
    EXP = Stim(stimf)

