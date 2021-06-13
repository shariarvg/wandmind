from django.apps import AppConfig
import html
import pathlib
import os

#import pandas as pd

import numpy as np
#import pandas as pd
import random

startPath = "/Users/shariarvaez-ghaemi/Desktop/wandmind/"

#import nltk

punctuation= '''!()-[]{};:'"\, <>./?@#$%^&*_~'''

#file_array=np.load("/Users/shariarvaez-ghaemi/icloud_drive/Documents/NLP_Demos/file_array.npy",allow_pickle=True)
new_pred = np.load("/Users/shariarvaez-ghaemi/icloud_drive/Documents/NLP_Demos/new_pred.npy",allow_pickle=True).item()
new_nlpred = np.load("/Users/shariarvaez-ghaemi/icloud_drive/Documents/NLP_Demos/new_nlpred.npy",allow_pickle=True).item()
pos_listW = np.load("/Users/shariarvaez-ghaemi/icloud_drive/Documents/NLP_Demos/pos_listW2.npy",allow_pickle=True).item()

'''
#Remove punctuation
keys = list(pos_listW.keys())
for ele in keys:
    for item in pos_listW[ele]:
        for p in punctuation:
            if p in item:
                pos_listW[ele].remove(item)
                break

'''

#new_pred = dict(enumerate(new_pred.flatten(), 1)) #Convert saved array to dictionary
#new_nlpred = dict(enumerate(new_nlpred.flatten(), 1))
#pos_listW = dict(enumerate(pos_listW.flatten(), 1))

def line_to_pos(line):
    thisLine = line
    thisLineS = ''
    for ele in thisLine:
        thisLineS += str(ele)+ " "
    thisLineTok = nltk.word_tokenize(thisLineS)
    thisLinePos = nltk.pos_tag(thisLineTok)
    thisLinePos = [thisLinePos[i][1] for i in range(len(thisLinePos))]
    return thisLinePos


def write_poetry(n_lines):

    n_lines = int(n_lines)

    lines = []
    for count in range(n_lines):
        gen_line_pos = []
        gen_line_words = []
        while len(gen_line_pos) < 6:
            tgl = tuple(gen_line_pos)
            possible_pos = new_pred[tgl]
            frequency_array = []
            for ele in list(possible_pos.keys()):
                prob = possible_pos[ele]
                for count in range(int(1000*prob)):
                    frequency_array.append(ele)
            newpos = random.choice(frequency_array)
            gen_line_pos.append(newpos)
            gen_line_words.append(random.choice(pos_listW[newpos]))
        lines.append(gen_line_words)

    for count in range(n_lines-1):
        thisLine = lines[count]
        thisLineS = ''
        for ele in thisLine:
            thisLineS += str(ele)+ " "
        thisLineTok = nltk.word_tokenize(thisLineS)
        thisLinePos = nltk.pos_tag(thisLineTok)
        earlier = [thisLinePos[i][1] for i in range(0,len(thisLinePos)-1)]
        last = thisLinePos[len(thisLinePos)-1][1]


        #Store lost versio
        subs = []
        
        while tuple(earlier) not in list(new_pred.keys()):
            earlier = earlier[1:-1]
        earlier = tuple(earlier)
        for pos in list(new_pred[earlier].keys()):
            #print(pos)
            prior = new_pred[earlier][pos]
            next3 = tuple(line_to_pos(lines[count+1])[0:3])
            if pos not in list(new_nlpred.keys()):
                posterior = 0
            elif next3 in list(new_nlpred[pos].keys()):
                posterior = new_nlpred[pos][next3]
            else:
                posterior = 0.0
            for c in range(int(1000000*prior*posterior)):
                subs.append(pos)
        #print(len(subs))
        if len(subs)>0:
            chosenPos = random.choice(subs)
        chosenWord = random.choice(pos_listW[pos])
        lines[count][-1]=chosenWord

    poemFinal = ""
    for line in lines:
        thisLine = ""
        for w in line:
            thisLine+=w
            thisLine+=" "
        poemFinal+=thisLine
        poemFinal+="<br>"


    return poemFinal


class WandmindConfig(AppConfig):
    name = 'wandmind'
    MODEL_PATH=pathlib.Path("model")
    PRETRAINED_PATH = pathlib.Path("model/wandmind_model.py")
    LABEL_PATH = pathlib.Path("label/")
    predictor = write_poetry
