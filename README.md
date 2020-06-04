# ExperimentNorming
Project for norming experimental stimuli relative to RNN models. Specifically, outputting 
surprisal metrics, frequency info, and checking if experimental vocabulary are contained 
in the models. 

### Dependencies
Requires the following python packages (available through pip):
* [pytorch](https://pytorch.org/) v1.0.0
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
There are two overarching options, outputing by word and by template. 

For by word output, the current version only works with two columns (with optional header) and multi_sent flag.

stimuli file
* expecting two columns with an optional header  

normed files are saved to results and can be formatted as an excel file or csv. The columns in this
are:
* SENT - Sentences combined from columns as given in stimuli file (lower-cased)
* UNK_SENT - Sentences combined with any missing vocab as \<unk\> (this is what the model sees)
* hasUNK - Boolean that is 0 if all words are in vocabulary, 1 otherwise
* WORD_ENTROPY_[MODEL] - Entropy after the word for the MODEL (one column per model)
* WORD_ENTROPY_AVG - Average entropy after the word across tested models
* WORD_REDUCTION_[MODEL] - Entropy reduction caused by the word for the MODEL (one column per model)
* WORD_REDUCTION_AVG - Average entropy reduction caused by the word across tested models
* WORD_SURP_[MODEL] - Surprisal at the word for the MODEL (one column per model)
* WORD_SURP_AVG - Average surprisal at the word across tested models

The outputted is padded so there is a consistent number of columns. If there are less words in a given 
sentence than the maximal, the word will be '' and the output values will be -1.


For template option, the specification is as follows:

stimuli file
* expecting one or two columns with an optional header with the sentences of the form DET NOUN (AUX)* VERB (PARTICLE) DET (ADJ |NOUN )\*NOUN 

normed files are saved to results and can be formatted as an excel file or csv. The columns in this
are:
* SENT1 - First sentence as given in the stimuli file (lower-cased)
* UNK_SENT1 - First sentence with any missing vocab as \<unk\> (this is what the model sees)
* hasUNK1 - Boolean that is 0 if all words are in vocabulary, 1 otherwise
* VERB1_ENTROPY_[MODEL] - Entropy after the verb for the MODEL (one column per model)
* VERB1_ENTROPY_AVG - Average entropy after the verb across tested models
* VERB1_REDUCTION_[MODEL] - Entropy reduction caused by the verb for the MODEL (one column per model)
* VERB1_REDUCTION_AVG - Average entropy reduction caused by the verb across tested models
* VERB1_SURP_[MODEL] - Surprisal at the verb for the MODEL (one column per model)
* VERB1_SURP_AVG - Average surprisal at the verb across tested models
* NOUN1_ENTROPY_[MODEL] - Entropy after the noun (head if modified) for the MODEL (one column per model)
* NOUN1_ENTROPY_AVG - Average entropy after the noun (head if modified) across tested models
* NOUN1_SURP_[MODEL] - Surprisal at the noun (head if modified) for the MODEL (one column per model)
* NOUN1_SURP_AVG - Average surprisal at the noun (head if modified) across tested models

Same thing repeats but for the second sentence and the flag is SENT2, VERB2, etc 
if there is a second column. In the case of stimuli linked as a discourse unit 
(--multi_sent) the second sentence measures are conditioned on the first sentence.

The directory vocab_info includes information about the frequency of words in 
the training corpora. Including: 
* raw_vocab - The vocabulary of the models
* tagged_vocab.csv - The vocabulary tagged for POS label
* noun_vocab_freq.csv - The nouns from the vocabulary with their frequency
* verb_vocab_freq.csv - The verbs from the vocabulary with their frequency

### Extra Details
The stimuli directory houses excel files with the data in the experiment. 

To run norm.py with non-default settings:
                
    usage: norm.py [-h] [--models MODELS] [--stim_file STIM_FILE] [--has_header]
                   [--multi_sent] [--template] [--output_file OUTPUT_FILE]
                   [--file_type FILE_TYPE]

    Experiment Stimuli Norming for LSTM Language Model Probing

    optional arguments:
      -h, --help            show this help message and exit
      --models MODELS       model to run [a|b|c|d|e|all]
      --stim_file STIM_FILE
                            path to stimuli file
      --has_header          Specify if the excel file has a header
      --multi_sent          Specify if you are running multiple sentence stimuli.
      --template            Specify if you want to use sentence template to focus
                            on verbs and nouns.
      --output_file OUTPUT_FILE
                            Ouput file name: default is normed_[stim_file_name]
      --file_type FILE_TYPE
                            File type for output: [xlsx|csv|both]

Example run:
        norm.py --models all --stim_file stimuli/multi_sent.xlsx --file_type xlsx --has_header --multi_sent

This will look for a stimuli file called multi_sent.xlsx that will have a header and that 
is specified to have the sentences in column one and two processed as a discourse unit
and output the stimuli RNN LM measures by word in an excel file 
in results called normed_multi_sent.xlsx.

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
