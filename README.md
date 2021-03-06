# ExperimentNorming
Project for norming experimental stimuli relative to RNN models. Specifically, outputting 
surprisal metrics, frequency info, computing similarity metrics, and checking if experimental vocabulary are contained 
in the models. 

### Dependencies
Requires the following python packages (available through pip):
* [pytorch](https://pytorch.org/) == v1.0.0
* [pandas](https://pandas.pydata.org) 
* [numpy](https://numpy.org)

To recreate the POS tags for the vocab you need:
  [spaCy](https://spacy.io) v2.2.4

From spaCy you need the pretrained English model "en_core_web_sm":

    python -m spacy download en_core_web_sm

If you want to run the other models (bert|gpt|tfxl) from bert.py:
* [allennlp](https://allennlp.org/) == v1.3.0
* [transformers](https://github.com/huggingface/transformers) == v3.0.0

This is much more limited at the moment so it may have to be tweaked (and may crash in non-use cases)


### Quick Usage
To run norming on stimuli:

    python norm.py --has_header --models all

This runs with default settings, stimuli file is assumed to be multi_sent.xlsx (in stimuli/), 
runs on all models (--models all), sentences will be treated seperately, and by-model and 
average output will be saved from IT experiment to results/normed_multi_sent(.xlsx and .csv). 

### Information on Files
There are five overarching options (or experiments), 
outputing by-word information-theoretic measures only (IT),
outputting by-word similarity to a baseline and information-theoretic measures (RSA),
one-shot learning a stimuli set and output change in surprisal (ADAPT), 
one-shot learning a stimuli set and output change in RSA (RSA-ADAPT), 
or checking a stimuli for UNKs (UNK). 

stimuli file
* expecting any number of columns, where each column is a sentence, with an optional header  
* if running RSA the first column will be treated as the baseline
* if running ADAPT the input format should be SENT1, ENT, SENT2, ENT
* if running RSA-ADAPT the input format should be BASELINE, COMPARISION, SENT1, ENT, BASELINE, COMPARISON, SENT2, ENT

normed files are saved to results and can be formatted as an excel file or csv. The columns in this
are, for each sentence (subscripted i, the baseline will be SENT_0 for RSA)
* SENT_i - Sentences combined from columns as given in stimuli file (lower-cased)
* UNK_SENT_i - Sentences combined with any missing vocab as \<unk\> (this is what the model sees)
* hasUNK_i - Boolean that is 0 if all words are in vocabulary, 1 otherwise

Then the columns range over each word by sentence. For mismatching sentence lengths a dummy word NULL with -1
values for the measures is appended:

* sent_i_word_k_ent_avg - Average entropy after the word across tested models
* sent_i_word_k_red_avg - Average entropy reduction caused by the word across tested models
* sent_i_word_k_surp_avg - Average surprsial at the word across tested models
* sent_i_word_k_sim_avg - Average similarity (if RSA) between the last hidden layer for the baseline and the word
* sent_i_word_k_ent_[MODEL] - Entropy after the word for the MODEL 
* sent_i_word_k_red_[MODEL] - Entropy reduction caused by the word for the MODEL
* sent_i_word_k_sup_[MODEL] - Surprisal at the word for the MODEL
* sent_i_word_k_sim_[MODEL] - Similarity (if RSA) between the last hidden layer for the baseline and the word for the MODEL


For ADAPT the columns are as follows:
* LOW - First column of sentences from input stimuli file
* LOW_ENT - Entropy of verb given in input stimuli file (second column in stimuli file)
* MODEL_delta - Change in surprisal at the final word after MODEL one-shot learns sentence
* avg_delta - Average change in surprisal at the final word after each model one-shot learns sentence
* HIGH - third column of sentences from input stimuli file
* HIGH_ENT - Entropy of verb given in input stimuli file (fourth column in stimuli file)
* MODEL_delta - Change in surprisal at the final word after MODEL one-shot learns sentence
* avg_delta - Average change in surprisal at the final word after each model one-shot learns sentence

For RSA-ADAPT the columns are as follows:
* BASELINE - First column from input stimuli, is baseline that similarity is calculated for
* COMPARISON - Second column from input stimuli, is sentence to compare similarity to baseline
* LOW - Third column of sentences from input stimuli file, sentences to learn from
* LOW_ENT - Entropy of verb given in input stimuli file (fourth column in stimuli file)
* MODEL_pre - Similarity prior to learning
* MODEL_post - Similarity after learning
* MODEL_diff - Difference between similarity after learning and before learning (positive means greater similarity after one-shot learning)
* avg_pre - Average similarity prior to learning sentences 
* avg_post - Average similarity after learning sentences 
* avg_diff - Average difference in similarity after learning sentences minus prior to learning 
* BASELINE - Fifth column from input stimuli, is baseline that similarity is calculated for
* COMPARISON - Sixth column from input stimuli, is sentence to compare similarity to baseline
* HIGH - Seventh column of sentences from input stimuli file, sentences to learn from
* HIGH_ENT - Entropy of verb given in input stimuli file (eighth column in stimuli file)
* MODEL_pre - Similarity prior to learning
* MODEL_post - Similarity after learning
* MODEL_diff - Difference between similarity after learning and before learning (positive means greater similarity after one-shot learning)
* avg_pre - Average similarity prior to learning sentences 
* avg_post - Average similarity after learning sentences 
* avg_diff - Average difference in similarity after learning sentences minus prior to learning 

The directory vocab_info includes information about the frequency of words in 
the training corpora. Including: 
* raw_vocab - The vocabulary of the models
* tagged_vocab.csv - The vocabulary tagged for POS label
* noun_vocab_freq.csv - The nouns from the vocabulary with their frequency
* verb_vocab_freq.csv - The verbs from the vocabulary with their frequency

### Extra Details
The stimuli directory houses excel files with the data in the experiment. 

To run norm.py with non-default settings:

    usage: norm.py [-h] [--exp EXP] [--models MODELS] [--vocab_file VOCAB_FILE]
                   [--has_header] [--multi_sent] [--avg] [--filter FILTER]
                   [--stim_file STIM_FILE] [--output_file OUTPUT_FILE]
                   [--file_type FILE_TYPE] [--cell_type CELL_TYPE] [--layer LAYER]

    Experiment Stimuli Norming for LSTM Language Model Probing

    optional arguments:
      -h, --help            show this help message and exit
      --exp EXP             experiment type [IT|RSA|ADAPT|RSA-ADAPT|UNK]
      --models MODELS       model to run
                            [a|b|c|d|e|all|big|web|bert|shuffled|elmo|gpt|tfxl]
      --vocab_file VOCAB_FILE
                            vocab file
      --has_header          Specify if the excel file has a header
      --multi_sent          Specify if you are running multiple sentence stimuli
                            (only for IT|RSA)
      --avg                 Specify if you want to return only average measures
      --filter FILTER       Specify name of file to words to filter from results
                            (only for IT|RSA)
      --stim_file STIM_FILE
                            path to stimuli file
      --output_file OUTPUT_FILE
                            Ouput file name: default is normed_[stim_file_name]
      --file_type FILE_TYPE
                            File type for output: [xlsx|csv|both|dill|cell]
      --cell_type CELL_TYPE
                            measure to output for the cell file type
      --layer LAYER         layer to get similarity at (for BERT, GPT2, TFXL)

Example run:
        
        norm.py --exp RSA --models all --stim_file stimuli/RSA_Analysis.xlsx --file_type xlsx --has_header --multi_sent --avg --filter

This will look for a stimuli file called RSA_Analysis.xlsx that will have a header and that 
has as its first column a baseline for RSA comparison. The remaining sentences will be 
processed as a discourse unit as per --multi_sent. The information-theoretic measures 
and similarity of the baseline to each of the words in the unit will be returned 
in an xlsx file called normed_RSA_Analysis.xlsx. By specifiying --models all and 
--avg, the output will be only the average values for that word across the models.  
Additionally running norm.py with --avg without
specifying and output file will append avg to the file name in results (normed_avg_RSA_Analysis.xlsx).

To run with the larger Wikipedia models or the Web models, download them from [here](https://zenodo.org/record/4053572#.X3OKrHaYU5k) and rename the directory
large_models and web_models, respectively. 

The flags filter and multi_sent only apply to the IT and RSA experiments. 

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

Included in main is a function adapt, which takes sentences from an excel file and one-shot
learns each sentence, returning surprisals before and after learning and the difference at
the target.

## Other Run Options
The file\_type dill will save the output as a binary file which can be used later (saved as instance of Stim class in data.py). The 
file\_type cell will save the output as stimuli sentence X model measure with the measure being specified with the cell\_type flag (use 
case is sim, but should work for ent, surp, red\_ent). Running non-lstm models (bert|gpt|tfxl|elmo) goes through bert.py (only 
validated for RSA experiments).

### References
