#!/bin/bash

if [ $# -ne 4 ]; then
    echo "usage: $0 TESTSET SRCLANG TGTLANG GEN"
    exit 1
fi

TESTSET=$1
SRCLANG=$2
TGTLANG=$3

GEN=$4

#if ! command -v sacremoses &> /dev/null
#then
#    echo "sacremoses could not be found, please install with: pip install sacremoses"
#fi

grep ^H $GEN \
| sed 's/^H\-//' \
| cut -f 3 \
| sacremoses detokenize \
> $GEN.sorted.detok
grep ^T $GEN \
| sed 's/^T\-//' \
| cut -f 2 \
| sacremoses detokenize \
> $GEN.sorted.ref.detok

sacrebleu --num-refs 1 $GEN.sorted.ref.detok --language-pair "${SRCLANG}-${TGTLANG}" < $GEN.sorted.detok