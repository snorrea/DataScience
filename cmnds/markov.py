# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 16:41:42 2016

let's make a markov bot!

@author: tower_000
"""
import re
import nltk.data


def splitMessage(mystring):
    sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s',mystring)
    #(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s
    #(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?)\s
    return sentences

    