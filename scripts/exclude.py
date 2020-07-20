#new_header = ['item', 'sent', 'hasUNK', 'verb_surp', 'verb_sim', 
        #'noun_surp', 'noun_sim'] 

remove_items = set([])

with open('yanina_flat_results.csv', 'r') as f:

    out_str = f.readline()

    for line in f:
        line = line.strip().split(',')
        if line[2] == '1':
            remove_items.add(line[0])
            continue
        elif line[0] in remove_items:
            continue

        line = ','.join(line) + '\n'
        out_str += line

with open('yanina_flat_results.csv', 'w') as f:

    f.write(out_str)
        
