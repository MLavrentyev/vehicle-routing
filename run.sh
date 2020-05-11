#!/bin/bash

########################################
############# CSCI 2951-O ##############
########################################
E_BADARGS=65
if [ $# -ne 1 ]
then
  echo "Usage: $(basename "$0") <input>"
  exit $E_BADARGS
fi

input=$1
output="output/$(basename -- "$input").sol"

#java -cp src solver.ls.Main $input

if hash pypy3 2>/dev/null; then
  #echo "using pypy3"
  if [[ "$OSTYPE" == "msys" ]]; then
    winpty pypy3 -W ignore src/solver.py "$input" -f "$output"
  else
    pypy3 -W ignore src/solver.py "$input" -f "$output"
  fi
elif hash pypy3.6-v7.1.1 2>/dev/null; then
  #echo "using pypy3.6-v7.1.1"
  pypy3.6-v7.1.1 -W ignore src/solver.py "$input" -f "$output"
else
  #echo "using python"
  python -W ignore src/solver.py "$input" -f "$output"
fi
