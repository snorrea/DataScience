from __future__ import print_function
import numpy as np
import random
import sys
import pandas as pd
from dateutil.parser import parse as dateParse

def initialCleanup(nameList,csvFile):
    tmp = pd.read_csv('_chat.txt', delimiter=':', header=None, names=headers, skiprows=2, dtype=dtypes, parse_dates=parseDates)
    
    df = pd.DataFrame()
    
    return tmp


def parseInput(filename):
    fin = open(filename,encoding='utf-8')
    array = []
    while True:
        line = fin.readline()
        if not line:
            break
        if len(line) < 22: #it is not a new message
            #print(array)
            array[-1][3] = array[-1][3]+" "+line
        elif is_date(line[0:10]):
            if line[22:].find(':')==-1:
                continue
            date = line[0:10]
            time = line[12:20]
            messagestarts = line[22:].find(':')+22 #where sender is
            sender = line[22:messagestarts]
            #print(str(messagestarts) + sender)
            message = line[messagestarts+1:-1]
            array.append([date, time, sender, message])
        else:
            array[-1][3] = array[-1][3]+" "+line
    fin.close()
    df = pd.DataFrame(array)
    df.columns = ["date","time","sender","message"]
    df['date'] = pd.to_datetime(df.date,dayfirst=True)
    return df
    

def parseInput2(filename):
    fin = open(filename,encoding='utf-8')
    array = []
    while True:
        line = fin.readline()
        if not line:
            break
        if len(line) < 22: #it is not a new message
            #print(array)
            array[-1][2] = array[-1][2]+" "+line
        elif is_date(line[0:10]):
            if line[22:].find(':')==-1:
                continue
            datetime = line[0:20]
            messagestarts = line[22:].find(':')+22 #where sender is
            sender = line[22:messagestarts]
            #print(str(messagestarts) + sender)
            message = line[messagestarts+1:-1]
            array.append([datetime, sender, message])
        else:
            array[-1][2] = array[-1][2]+" "+line
    fin.close()
    df = pd.DataFrame(array)
    df.columns = ["datetime","sender","message"]
    df['datetime'] = pd.to_datetime(df.datetime,dayfirst=True)
    return df


def is_date(string):
    try: 
        dateParse(string)
        #print(string + " is a date")
        return True
    except ValueError:
        #print(string + " is not a date")
        return False