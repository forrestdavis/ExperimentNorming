from main import *
import argparse

parser = argparse.ArgumentParser(description='Experiment Stimuli Norming for LSTM Language Model Probing')

parser.add_argument('--exp', type=str, default='IT', 
                    help='experiment type [IT|RSA|ADAPT|RSA-ADAPT]')

parser.add_argument('--models', type=str, default='a',
                    help='model to run [a|b|c|d|e|all]')

parser.add_argument('--has_header', action='store_true',
                    help='Specify if the excel file has a header')

parser.add_argument('--multi_sent', action='store_true',
                    help='Specify if you are running multiple sentence stimuli (only for IT|RSA)')

parser.add_argument('--avg', action='store_true', 
                    help='Specify if you want to return only average measures')

parser.add_argument('--filter', type=str, 
        default=None, 
        help='Specify name of file to words to filter from results (only for IT|RSA)')

parser.add_argument('--stim_file', type=str, 
        default='stimuli/multi_sent.xlsx', 
        help='path to stimuli file')

parser.add_argument('--output_file', type=str, 
        default='', 
        help='Ouput file name: default is normed_[stim_file_name]')

parser.add_argument('--file_type', type=str, 
        default='both', 
        help='File type for output: [xlsx|csv|both]')

args = parser.parse_args()

#hard code data_dir
data_path = './'
#hardcode vocab file for now
#vocab_file = '/Users/forrestdavis/Data/models/vocab'

#set device to gpu for work on desktop :)
#device = torch.device('cuda:0')

#set device to cpu for work on laptop
device = torch.device('cpu')

#set loss function to be cross entropy
criterion = nn.CrossEntropyLoss()

#Get pretrained model files
model_files = glob.glob('./models/*.pt')
if args.models == 'a':
    model_files = list(filter(lambda x: '_a_' in x, model_files))[:1]
elif args.models == 'b':
    model_files = list(filter(lambda x: '_b_' in x, model_files))[:1]
elif args.models == 'c':
    model_files = list(filter(lambda x: '_c_' in x, model_files))[:1]
elif args.models == 'd':
    model_files = list(filter(lambda x: '_d_' in x, model_files))[:1]
elif args.models == 'e':
    model_files = list(filter(lambda x: '_e_' in x, model_files))[:1]

model_files.sort()

vocab_file = 'models/vocab'
verbose = True

if args.exp == 'IT':
    hasSim = False
else:
    hasSim = True

if args.output_file is '':
    if args.avg:
        output_file = 'results/normed_avg_'+args.stim_file.split('/')[-1]
    else:
        output_file = 'results/normed_'+args.stim_file.split('/')[-1]
else:
    output_file = 'results/'+args.output_file

#Run experiment
if args.exp == 'IT':
    EXP = run_norming(args.stim_file, vocab_file, model_files, args.has_header,
            args.multi_sent, args.filter, verbose)
elif args.exp == 'RSA':
    EXP = run_RSA(args.stim_file, vocab_file, model_files, args.has_header, 
            args.multi_sent, args.filter, verbose)

elif args.exp == 'ADAPT':
    run_adapt(args.stim_file, vocab_file, model_files, output_file, args.has_header, 
            args.avg, args.file_type)

elif args.exp == 'RSA-ADAPT':
    run_RSA_adapt(args.stim_file, vocab_file, model_files, output_file, args.has_header, 
            args.avg, args.file_type)

#save output IT|RSA
if args.exp == 'IT' or args.exp == 'RSA':

    if args.file_type == 'both':
        EXP.save_excel(output_file, model_files, args.avg, hasSim)
        EXP.save_csv(output_file, model_files, args.avg, hasSim)
    elif args.file_type == 'csv':
        EXP.save_csv(output_file, model_files, args.avg, hasSim)
    elif args.file_type == 'xlsx':
        EXP.save_excel(output_file, model_files, args.avg, hasSim)
