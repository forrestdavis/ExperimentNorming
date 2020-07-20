
targets = []

with open('yanina_full_targets.csv', 'r') as f:

    header = f.readline().strip().split(',')

    for line in f:
        line = line.strip().split(',')

        targets.append(line)

ratings = []
with open('yanina_flat_ratings.csv', 'r') as f:
    header = f.readline()

    for line in f:
        line = line.strip().split(',')

        ratings.append(line)




new_header = ['item', 'sent', 'hasUNK', 'rating', 'cond', 'model', 'verb_surp', 'verb_sim', 
        'noun_surp', 'noun_sim'] 

exclude_items = set([])

count = 0
out_str = ','.join(new_header)+'\n'

models = []
with open('results/normed_yanina_full.csv', 'r') as data:



    header = data.readline().strip().split(',')
    
    for elem in header:
        if 'LSTM' in elem:
            m = elem.split('_')[-1].split('-')[0]
            if m not in models:
                models.append(m)

    for line in data:
        line = line.strip().split(',')

        item = (count % 160) + 1
        hasUNK = int(line[4]) + int(line[5])
        if hasUNK > 1:
            hasUNK = 1
        sent = line[1]

        ts = targets[count]

        y = 6
        repeat = [item, sent, hasUNK, ratings[count][0], ratings[count][1]]
        verb_surp = [-1]*len(models)
        verb_sim = [-1]*len(models)
        noun_surp = [-1]*len(models)
        noun_sim = [-1]*len(models)
        while y < len(line):

            head = header[y].split('_')
            if len(head) == 4:
                word = line[y]

                y += 1

                head = header[y].split('_')
                while head[4] in ['surp', 'sim', 'ent', 'red']:

                    if word == ts[0]: 
                        if 'avg' not in head[5]:
                            if 'surp' in head[4]:
                                index = models.index(head[-1].split('-')[0])
                                verb_surp[index] = line[y]
                                verb_sim[index] = line[y+1]

                    elif word == ts[1]: 
                        if 'avg' not in head[5]:
                            if 'surp' in head[4]:
                                index = models.index(head[-1].split('-')[0])
                                noun_surp[index] = line[y]
                                noun_sim[index] = line[y+1]

                    y += 1
                    if y == len(line):
                        break

                    head = header[y].split('_')
                    if len(head) == 4:
                        break

        count += 1

        for z in range(len(models)):
            data = repeat + [models[z], verb_surp[z], verb_sim[z], noun_surp[z], noun_sim[z]]
            data = list(map(lambda x: str(x), data))
            out_str += ','.join(data) + '\n'


with open('yanina_flat_results.csv', 'w') as o:
    o.write(out_str)
