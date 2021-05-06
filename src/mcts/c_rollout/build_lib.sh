#! /bin/bash
# Instructions following https://nesi.github.io/perf-training/python-scatter/ctypes
# Can also run g++ -std=c++20 -o mylib.so -shared rollout.cpp as is done here
# https://stackoverflow.com/questions/30983220/ctypes-error-attributeerror-symbol-not-found-os-x-10-7-5

python setup.py build
