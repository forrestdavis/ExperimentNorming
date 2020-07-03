# ExperimentNorming
Project for norming experimental stimuli relative to RNN models. Specifically, outputting 
surprisal metrics, frequency info, and checking if experimental vocabulary are contained 
in the models. 

### Dependencies
Requires the following python packages (available through pip):
* [pytorch](https://pytorch.org/) >= v1.0.0
* [pandas](https://pandas.pydata.org) 
* [numpy](https://numpy.org)

To recreate the POS tags for the vocab you need:
  [spaCy](https://spacy.io) v2.2.4

From spaCy you need the pretrained English model "en_core_web_sm":

    python -m spacy download en_core_web_sm


### Quick Usage
To run norming on stimuli:

    python norm.py --models all

### Information on Files
There are two overarching options, outputing by information-theoretic measures only (IT) and
outputting similarity to a baseline and information-theoretic measures (RSA). Individual 
sentences should be placed in seperate columns. 

stimuli file
* expecting any number of columns, where each column is a sentence, with an optional header  
* if running RSA the first column will be treated as the baseline

normed files are saved to results and can be formatted as an excel file or csv. The columns in this
are, for each sentence (subscripted i, the baseline will be SENT_0 for RSA)
* SENT_i - Sentences combined from columns as given in stimuli file (lower-cased)
* UNK_SENT_i - Sentences combined with any missing vocab as \<unk\> (this is what the model sees)
* hasUNK_i - Boolean that is 0 if all words are in vocabulary, 1 otherwise

Then the columns range over each word by sentence. For mismatching sentence lengths a dummy word NULL with -1
values for the measures is appended:

* sent_i_word_j_ent_avg - Average entropy after the word across tested models
* sent_i_word_j_red_avg - Average entropy reduction caused by the word across tested models
* sent_i_word_j_surp_avg - Average surprsial at the word across tested models
* sent_i_word_j_sim_avg - Average similarity (if RSA) between the last hidden layer for the baseline and the word
* sent_i_word_j_ent_[MODEL] - Entropy after the word for the MODEL 
* sent_i_word_j_red_[MODEL] - Entropy reduction caused by the word for the MODEL
* sent_i_word_j_sup_[MODEL] - Surprisal at the word for the MODEL
* sent_i_word_j_sim_[MODEL] - Similarity (if RSA) between the last hidden layer for the baseline and the word for the MODEL

The directory vocab_info includes information about the frequency of words in 
the training corpora. Including: 
* raw_vocab - The vocabulary of the models
* tagged_vocab.csv - The vocabulary tagged for POS label
* noun_vocab_freq.csv - The nouns from the vocabulary with their frequency
* verb_vocab_freq.csv - The verbs from the vocabulary with their frequency

### Extra Details
The stimuli directory houses excel files with the data in the experiment. 

To run norm.py with non-default settings:
                
    usage: norm.py [-h] [--exp EXP] [--models MODELS] [--has_header]
                   [--multi_sent] [--avg] [--filter FILTER]
                   [--stim_file STIM_FILE] [--output_file OUTPUT_FILE]
                   [--file_type FILE_TYPE]

    Experiment Stimuli Norming for LSTM Language Model Probing

    optional arguments:
      -h, --help            show this help message and exit
      --exp EXP             experiment type [IT|RSA]
      --models MODELS       model to run [a|b|c|d|e|all]
      --has_header          Specify if the excel file has a header
      --multi_sent          Specify if you are running multiple sentence stimuli
      --avg                 Specify if you want to return only average measures
      --filter FILTER       Specify name of file to words to filter from results
      --stim_file STIM_FILE
                            path to stimuli file
      --output_file OUTPUT_FILE
                            Ouput file name: default is normed_[stim_file_name]
      --file_type FILE_TYPE
                            File type for output: [xlsx|csv|both]


Example run:
        norm.py --exp RSA --models all --stim_file stimuli/RSA_Analysis.xlsx --file_type xlsx --has_header --multi_sent --avg --filter

This will look for a stimuli file called RSA_Analysis.xlsx that will have a header and that 
has as its first column a baseline for RSA comparison. The remaining sentences will be 
processed as a discourse unit as per --multi_sent. The information-theoretic measures 
and similarity of the baseline to each of the words in the unit will be returned 
in an xlsx file called normed_RSA_Analysis.xlsx. By specifiying --models all and 
--avg, the output will be only the average values for that word across the models.  
Additionally running norm.py with --avg without
specifying and output file will append avg to the file name in results.

To recreate the frequency count information in vocab_info, you need 
the training corpora for the models. Unfortunately, they
are too large to be pushed to git. They will be added 
to a Zenodo archive if this is published. Otherwise, you can email me for 
it. The pipeline for when you have the corpora is to first get POS tags
for the vocab. 

    python scripts/tag.py models/vocab

This will saved the tag vocab to vocab_info as tagged_vocab.csv. Next 
you need to get the word frequency counts for the corpora. To do this:

    ./scripts/get_counts.sh [a-e] > OUTPUT_FNAME

The results from running this on my end are in vocab_info/[A-E]_word_freqs.
As you can see, the output is sorted from most to least frequent word 
in the corpus.

Given these files, you can run freq.py to get the verb and noun frequencies
csv files, which are saved in vocab_info as noun_vocab_freq.csv and verb_vocab_freq.csv. 

    python scripts/freq.py

### References
