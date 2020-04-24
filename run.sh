#!/bin/bash

########################################
############# CSCI 2951-O ##############
########################################
E_BADARGS=65
if [ $# -ne 1 ]
then
	echo "Usage: `basename $0` <input>"
	exit $E_BADARGS
fi
	
input=$1

#java -cp src solver.ls.Main $input
if hash pypy3 2>/dev/null; then
	echo "using pypy3"
	pypy3 src/main.py $input
elif hash pypy3.6-v7.1.1 2>/dev/null; then
	echo "using pypy3.6-v7.1.1"
	pypy3.6-v7.1.1 src/main.py $input
else
	echo "using python"
	python src/main.py $input
fi