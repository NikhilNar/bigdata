#!/usr/bin/env python
"""A more advanced Mapper, using Python iterators and generators."""

import sys
import re
import os
import io



def read_input(file):
    for line in file:
        # split the line into words
        line = re.sub('([^\s\w]|_)+', '', line).lower()
        yield line.split()
        

# def read_input(path):
#     print("path=", path)
#     for filename in os.listdir(path):
#         with open(path+filename, "r") as f:
#             for line in f.readlines():
#                 # split the line into words
#                 line = re.sub('([^\s\w]|_)+', ' ', line).lower()
#                 yield line.split()


def main(separator='\t'):
    # input comes from STDIN (standard input)
    data = read_input(sys.stdin)
    #data = read_input(sys.argv[1])
    for words in data:
        # write the results to STDOUT (standard output);
        # what we output here will be the input for the
        # Reduce step, i.e. the input for reducer.py
        #
        # tab-delimited; the trivial word count is 1
        wordSize = len(words)
        if wordSize == 1:
            continue
        for i in range(0, wordSize-1):
            print('{}{}{}'.format(words[i]+" "+words[i+1], separator, 1))


if __name__ == "__main__":
    main()
