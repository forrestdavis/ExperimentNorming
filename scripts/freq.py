import subprocess
import stimuli
import sys


#Load in freq info
freqs = {}
files = ['vocab_info/A_word_freqs', 
        'vocab_info/B_word_freqs',
        'vocab_info/C_word_freqs',
        'vocab_info/D_word_freqs',
        'vocab_info/E_word_freqs']

for f in files:
    corpus = f.split('/')[1].split('_')[0]
    #freqs[corpus] = {}
    with open(f, 'r', errors='ignore') as data:
        for line in data:
            line = line.strip().split(' ')
            if len(line) != 2:
                continue
            freq = line[0]
            word = line[1]
            if word not in freqs:
                freqs[word] = []
            freqs[word].append(freq)

verbs = {}
nouns = {}

fname = 'vocab_info/tagged_vocab.csv'

HEADER = 'word,pos,freqA,freqB,freqC,freqD,freqE\n'
noun_outf = open('vocab_info/noun_vocab_freq.csv', 'w')
verb_outf = open('vocab_info/verb_vocab_freq.csv', 'w')

noun_outf.write(HEADER)
verb_outf.write(HEADER)

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
                if word in freqs:
                    out_str = ','.join([word, pos] + freqs[word])+'\n'
                    noun_outf.write(out_str)
            elif "VB" in pos:
                if word in freqs:
                    out_str = ','.join([word, pos] + freqs[word])+'\n'
                    verb_outf.write(out_str)

noun_outf.close()
verb_outf.close()
