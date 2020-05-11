#!/bin/bash

# $2 is word to check $1 is corpus [a-e]

corpus="corpora/wiki103_80m_${1}.train"

#if particle combine
if [ -n "$4" ]; then
    word="${2} ${3} ${4}"
else
    if [ -n "$3" ]; then
    word="${2} ${3}"
    else
        word=$2
    fi
fi

out=$(cat $corpus | grep -ow "${word}" | wc -l)

echo $out
