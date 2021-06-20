

# Wandering Mind: Teaching A Statistical Language Model to Write Poetry


"MacHIne LeARniNG caN'T bE crEAtIve." <br>

We've heard it over and over again. No matter how many strides we can make in data science, we will never teach a computer to be the next Picasso, Shakespeare, or Drake. <br>

But what if we <i>could</i> teach an ML to be creative? More specifically, what if we could feed thousands of poems to an ML, have it learn basic conventions like syntax and grammar, and allow it to scatter its thoughts about? <br>

Today, we will be leveraging a Monte-Carlo simulation with POS tagging to write original poems. <b> We will be writing our code in Python and using Django to deploy it as a local website.</b><br>


Author: Shariar Vaez-Ghaemi <br>
Date: 6/4/21 <br>

### The Data

We will be using <a href = "https://github.com/aparrish/gutenberg-poetry-corpus"> the Gutenberg Poetry dataset.</a><br>   Each line in the corpus is a JSON object. The developer – Allison Parrish – does an excellent job of explaining the usability and mutability of the package in her README, so check it out!


```python
#Data Source: Project Gutenberg

import gzip, json, string
all_lines = []
for line in gzip.open("/Users/shariarvaez-ghaemi/Downloads/gutenberg-poetry-v001.ndjson.gz"):
    all_lines.append(json.loads(line.translate(str.maketrans('', '', string.punctuation))))

```

Next, we'll import NLTK, which we will be using to tag the part-of-speech for every word in every line of the corpus. Eventually, we will use these part-of-speech tags as training data to teach syntax sequences to our ML, but for now, let's see how NLTK will tokenize and POS-tag the first line of Robert Frost's notorious "Fire and Ice."


```python
#Practice: Tokenizing and Part-of-Speech (POS) tagging on first line
import nltk

firstLine = "Some say the world will end in fire."
print(firstLine)
tokens = nltk.word_tokenize(firstLine) #Tokenizer
print("Tokens from first line: ",tokens)

pos_tags = nltk.pos_tag(tokens)
print("Parts of Speech: ",pos_tags)
```

    Some say the world will end in fire.
    Tokens from first line:  ['Some', 'say', 'the', 'world', 'will', 'end', 'in', 'fire', '.']
    Parts of Speech:  [('Some', 'DT'), ('say', 'VBP'), ('the', 'DT'), ('world', 'NN'), ('will', 'MD'), ('end', 'VB'), ('in', 'IN'), ('fire', 'NN'), ('.', '.')]


Family Reunion Time: Let's unite all lines of each poem into an array. Every element of the array will be a multi-line poem.


```python
#Make each poem an array of lines
#Make the corpus an array of poems

punctuation= '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
emptyString=""

file_array = []
i_file = []
ticker = 0
for line in all_lines:
    words = line['s']
    for p in punctuation:
        words.replace(p,emptyString)
    if line['gid'] == str(ticker):
        i_file.append(words)
    else:
        file_array.append(i_file)
        i_file = []
        i_file.append(words)
        ticker = line['gid']

import numpy as np
np_fa = np.asarray(file_array)
np.savetxt("file_array.csv",np_fa,delimiter='  ', header='string', comments='', fmt='%s')
```

### Training

Now we're at the beginning of the training process. We're going to create a predictive dictionary of part-of-speech sequences. Each key will be an <i> ordered tuple</i> containing a part-of-speech sequence (e.g. verb-adverb-noun), while each value will be an <i> unordered array</i> containing all the parts-of-speech that came directly after it. Yes, there will be repeats in the value array. <br>

While we're doing this, we will also build a dictionary with each key being a POS and each value being an array of examples for that POS. For example, we may have an entry like: <br>

{JJ:["funny","loud","annoying"]} <br>

We need this for poem generation, too.


```python
#Build Predictive Dictionary of Part-of-Speech Sequences

predW = {} #{[[Known_Pos_1, Known_Pos_2, Known_Pos_3]]:[[Next_Pos_A],[Next_Pos_B]]}
pos_listW = {}
counter = 0
for line in all_lines:
    line_tokens = nltk.word_tokenize(line['s'])
    line_pos = nltk.pos_tag(line_tokens)
    known_pos = []
    for ele in line_pos:
        word = ele[0] #Word
        p = ele[1] #POS
        #---
        #Add word to dictionary of examples of POS
        if p in pos_listW.keys():
            pos_listW[p].append(word)
        else:
            pos_listW[p]=[]
            pos_listW[p].append(word)
        #---
        #Add POS to POS sequencing dictionaries
        tup_kp = tuple(known_pos)
        if tup_kp in predW.keys():
            predW[tup_kp].append(p) 
            known_pos.append(p)
        else:
            predW[tup_kp]=[]
            predW[tup_kp].append(p)
            known_pos.append(p)
```

Now that we're done with the dictionary, we'll realize that each entry looks something like this: <br>

[JJ,NNP]:[JJ,JJ,NNP,RB,RB,RB,JJ,RB] <br>

This will work great when we're trying to sample from the value array to write the poem. However, it won't work great when we're trying to interpret the value array as a frequency distribution. What we'd rather have is: <br>

[JJ,NNP]:{JJ:0.375,NNP:0.125,RB:0.5} <br>

Where each key in the value dictionary is a possible next-POS, and each value in the value dictionary is the likelihood of seeing that POS. The function freq_dict will help us transform the data structure:


```python
#Turn an array of elements into a dictionary of {element: frequency}
#Before: {x:[a,a,b,c,]}
#After: {x:{a:0.5,b:0.25,c:0.25}}

def freq_dict(d):
    new_d = {}
    keys = list(d.keys())
    for key in keys:
        entry = d[key]
        counts = {}
        total = 0
        for ele in entry:
            total+=1
            if (type(ele) is list):
                ele = tuple(ele)
            if ele in list(counts.keys()):
                counts[ele]+=1
            else:
                counts[ele]=1
        for index in list(counts.keys()):
            counts[index] = float(counts[index])/total
        new_d[key]=counts
    return new_d
```


```python
new_pred = freq_dict(predW)
```

In addition to predicting the next word in a line, we also want to predict how a new line of the poem will start given the last word of the old line. This will help us weave the lines together grammatically. <br>

Here, we will take the frequency of next-3-POS sequences given the last POS in a line.


```python
#New Dictionary. Key = Last word of line. Value = All observed three-word sequences at the start of the next line

import numpy as np

#file_array = np.load("file_array.csv",allow_pickle=True)

nextLinePred = {}

for poem in file_array:
    length = len(poem)
    for i in range(length-1):
        line1 = poem[i]
        line2 = poem[i+1]
        tl1 = nltk.word_tokenize(line1)
        tl2 = nltk.word_tokenize(line2)
        pos1 = nltk.pos_tag(tl1)
        pos2 = nltk.pos_tag(tl2)
        finalPos1 = pos1[-1][1] #Last POS from first line
        threePos2 = [ele[1] for ele in pos2[0:3]] #First 3 POS from second line
        if (finalPos1 in list(nextLinePred.keys())):
            nextLinePred[finalPos1].append(threePos2)
        else:
            nextLinePred[finalPos1]=[]
            nextLinePred[finalPos1].append(threePos2)
```


```python
#Change dictionary value from observation array to frequency dictionary

new_nlpred = freq_dict(nextLinePred)
```

This next function will simply help us transform an array of words into an array of POS's. This will be crucial when our poem generator analyzes what words it has written so-far in order to identify the next word to write.


```python
#Turn an array of words into an array of parts of speech

def line_to_pos(line):
    thisLine = line
    thisLineS = ''
    for ele in thisLine:
        thisLineS += str(ele)+ " "
    thisLineTok = nltk.word_tokenize(thisLineS)
    thisLinePos = nltk.pos_tag(thisLineTok)
    thisLinePos = [thisLinePos[i][1] for i in range(len(thisLinePos))]
    return thisLinePos
```
### Poetry Generation

We're finally at the point where we get to write some poetry. Here is the algorithm we will use: <br>

<ol>
    <li>Given the POS's written in the line so far, use the POS-sequence dictionary to randomly sample the next POS. Do so until you get to the end of the line. Then, write the next line in the same way.</li>
    <li>Once we're done with all lines, go back and check the last word of each line. Does the last POS of each line flow into the first three POS's of each line? We'll use Bayes Theorem to identify the likelihood of seeing the current POS as the last POS in each line, given the prior probability (from the POS-sequence dictionary) and the probability of seeing the first three words of the next line.</li>
</ol>



```python
#Bayesian Poetry Generation

def write_poetry(n_lines):

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
                subs.append(word)
        #print(len(subs))
        chosenPos = random.choice(subs)
        chosenWord = random.choice(pos_listW[pos])
        lines[count][-1]=chosenWord
    
    for line in lines:
        for w in line:
            print(w,end=" ")
        print("\n")
```

Let's write some poetry! How about we start with 5 lines?


```python
import random
write_poetry(5)
```

### Saving Files for Django Backend

If you're interested in deploying this model on a website, save the existing Numpy arrays into the same project folder that you download this repository into it.We certainly don't want to re-train the model everytime we load our website, so we'll just load our dictionaries. <br>

This repository already contains all the Django files you need to deloy your model. All you need to do is set up the directory in your terminal and run the following code: <br>
<code> python manage.py runserver </code>


```python
np.save("file_array.npy",np.asarray(file_array),allow_pickle=True)
```


```python
np.save("new_pred.npy",np.asarray(new_pred))
```


```python
np.save("new_nlpred.npy",np.asarray(new_nlpred))
```


```python
np.save("pos_listW.npy",np.asarray(pos_listW))
```
