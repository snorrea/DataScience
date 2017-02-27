from __future__ import print_function
import numpy as np
import random
import sys
import pandas as pd
import pylab as P
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Activation, Dropout, LSTM
from keras.optimizers import Nadam, SGD
import support as sup
import markov

nameList = [['Erik Bjørklund', '§ '], 
          ['Christian Selen', '| '], 
          ['Snorre', '¤ '], 
          ['Erlend Dybvig Sørmoen', '# '], 
          ['Øivind Bjørklund', '$ '],
          ['Anders Bjørklund', '€ '] ]
#inputArr = sup.initialCleanup(nameList,"_chat.txt")
df = sup.parseInput2("_chat.txt") #["date","time","sender","message"]


#plot number of messages per day
#messageByDays = df.groupby(['date']).count()

#df['date'].groupby([df.date.dt.year, df.date.dt.month, df.date.dt.week]).count().plot(kind="bar",grid=True, figsize=(20,10))
#df['date'].groupby([df.date.dt.year, df.date.dt.month]).count().plot(kind="bar",grid=True, figsize=(20,10))
#df['datetime'].groupby([df.datetime.dt.year, df.datetime.dt.month]).count().plot(kind="bar",grid=True, figsize=(20,10))
#, subplots=True
#df['time'].grouby
#P.show()
df['datetime'].groupby([df.datetime.dt.weekday]).count().plot(kind="bar",grid=True, figsize=(20,10))


#plot number of messages per time per day

#plot number of words per sender

