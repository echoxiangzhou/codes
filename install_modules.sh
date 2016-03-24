#!/bin/bash

here=$(pwd) 

echo ' '
echo '#- update the following files :'
ls -1 $here/*.py 
cp $here/*.py   $HOME/lib/python/

