import subprocess
import stimuli
import sys


def get_freq(word, model='a'):

    count = subprocess.getoutput("./freq.sh %s %s" %(model, word))
    return count

verbs = set([])
nouns = set([])

fname = 'vocab_info/tagged_vocab.csv'

#Get nouns and verbs
with open(fname, 'r') as f:
    #skip header
    f.readline()
    for line in f:
        line = line.strip().split('",')
        if len(line) > 1:
            word = line[0].replace('"', '')
            pos = line[1]
            if 'NN' in pos:
                nouns.add(word +' '+ pos)
            elif "VB" in pos:
                verbs.add(word + ' ' + pos)

models = ['a', 'b', 'c', 'd', 'e']

print('Getting noun frequency info...')
with open('vocab_info/noun_vocab_freq.csv', 'w') as f:
    HEADER = 'word,pos,freqA,freqB,freqC,freqD,freqE\n'
    f.write(HEADER)
    NOUNs = {}
    for model in models:
        for noun in nouns:
            pos = noun.split(' ')[1]
            noun = noun.split(' ')[0]
            freq = get_freq(noun, model)
            if noun not in NOUNs:
                NOUNs[noun] = [pos]
            NOUNs[noun].append(str(freq))

    #Write to output
    for noun in NOUNs:
        outstr = noun+','+','.join(NOUNs[noun])+'\n'
        f.write(outstr)

print('Getting verb frequency info...')
with open('vocab_info/verb_vocab_freq.csv', 'w') as f:
    HEADER = 'word,pos,freqA,freqB,freqC,freqD,freqE\n'
    f.write(HEADER)
    VERBs = {}
    for model in models:
        for verb in verbs:
            pos = verb.split(' ')[1]
            verb = verb.split(' ')[0]
            freq = get_freq(verb, model)
            if verb not in VERBs:
                VERBs[verb] = [pos]
            VERBs[verb].append(str(freq))

    #Write to output
    for verb in VERBs:
        outstr = verb+','+','.join(VERBs[verb])+'\n'
        f.write(outstr)

