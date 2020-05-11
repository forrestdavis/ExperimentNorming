# ExperimentNorming
Project for norming experimental stimuli relative to RNN models. Specifically, outputting 
surprisal metrics, frequency info, and checking if experimental vocabulary are contained 
in the models. 

### Dependencies
Requires the following python packages (available through pip):
* [pytorch](https://pytorch.org/) v1.0.0

To recreate the POS tags for the vocab you need:
  [spaCy](https://spacy.io) v2.2.4

From spaCy you need the pretrained English model "en_core_web_sm":

  python -m spacy download en_core_web_sm


### Quick Usage
To run norming on stimuli:


### Extra Details
The stimuli directory houses excel files with the data in the experiment. 

To recreate the frequency count information in vocab_info, you need 
the training corpora for the models. Unfortunately, they
are too large to be pushed to git. They will be added 
to a Zenodo archive if this is published. Otherwise, you can email me for 
it. 


Fuller details to be added later

### References
