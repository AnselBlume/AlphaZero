#! /bin/zsh

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -r
