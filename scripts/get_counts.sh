#!/bin/bash

# $2 is word to check $1 is corpus [a-e]

corpus="corpora/wiki103_80m_${1}.train"

awk -f word_frequencies.awk $corpus | sort -r -s -n -k 1,1
