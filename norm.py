from main import *
import argparse

parser = argparse.ArgumentParser(description='Experiment Stimuli Norming for LSTM Language Model Probing')

parser.add_argument('--models', type=str, default='a',
                    help='model to run [a|b|c|d|e|all]')

parser.add_argument('--stim_file', type=str, 
        default='stimuli/The_boy_will_bounce_the_ball.xlsx', 
        help='path to stimuli file')

parser.add_argument('--has_header', action='store_true',
                    help='Specify if the excel file has a header')

parser.add_argument('--multi_sent', action='store_true',
                    help='Specify if you are running multiple sentence stimuli.')

parser.add_argument('--template', action='store_true', 
                    help='Specify if you want to use sentence template to focus on verbs and nouns.')

parser.add_argument('--avg', action='store_true', 
                    help='Specify if you want to return avgerage measures only (only works for default by-word measures).')

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


#set device to cpu for work on desktop :/
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

vocab_file = 'models/vocab'

#Run experiment
EXP = run_norming(args.stim_file, vocab_file, model_files, 
        args.has_header, args.multi_sent, 
        args.template, True)

if args.output_file is '':
    if args.avg:
        output_file = 'results/normed_avg_'+args.stim_file.split('/')[-1]
    else:
        output_file = 'results/normed_'+args.stim_file.split('/')[-1]
else:
    output_file = 'results/'+args.output_file

if args.file_type == 'both':
    EXP.save_excel(output_file, args.avg)
    EXP.save_csv(output_file, args.avg)
elif args.file_type == 'csv':
    EXP.save_csv(output_file, args.avg)
elif args.file_type == 'xlsx':
    EXP.save_excel(output_file, args.avg)
