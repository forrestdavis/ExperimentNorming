import spacy
import sys
import os

nlp = spacy.load("en_core_web_sm")

data = open(sys.argv[1], 'r')

name = os.path.split(sys.argv[1])[-1]


out = open('vocab_info/tagged_'+name+'.csv', 'w')
out.write('word, pos\n')
for sent in data:
    sent = sent.strip()
    #propagate special
    if sent == "<unk>" or sent == "<num>" or sent == "<eos>":
        out.write(sent + "," + "NA\n")
        words.append(sent)
        continue
    sent_info = nlp(sent.lower())
    words = []
    for token in sent_info:
        words.append(token.text)
        words.append(token.tag_)

    mod_sent = ','.join(words)  
    out.write('"'+words[0]+'"'+','+words[1]+'\n')

out.close()
data.close()
