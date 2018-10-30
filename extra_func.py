import numpy as np
import tensorflow as tf
import argparse
import os
from six.moves import cPickle
import nltk
import time
import queue as Q
from operator import itemgetter
from model import Model
import itertools
from numpy.random import choice
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors
import string
import re
import collections
import sys
import pickle
import getopt

from nltk.corpus import stopwords
import random


def isFitPattern(pattern,stress_num):
    if(len(pattern)+stress_num>10):
        return False
    i = stress_num
    ind = 0
    while(ind<len(pattern)):
        if(stress_num%2 == 0):
            if(pattern[ind]!="0"):
                return False
        else:
            if(pattern[ind]!="1"):
                return False
        ind+=1
        stress_num+=1
    return True

def createMeterGroups(corpus,dictMeters):
    ret = {}
    for word in corpus:
        if(word not in dictMeters):
            continue #THIS IS KEY STEP
        for pattern in dictMeters[word]: #THIS GENERATES REPEAT INSTANCES OF WORDS ter
            if(pattern not in ret):
                ret[pattern] = set([word])
            else:
                ret[pattern].add(word)
    return ret

def createPartSpeechTags(corpus,dictMeters):
    dictPartSpeechTags = {}
    for word in corpus:
        if(word not in dictMeters):
            continue
        token = nltk.word_tokenize(word)
        tag = nltk.pos_tag(token)
        dictPartSpeechTags[word] = tag[0][1]
    return dictPartSpeechTags

#DICTIONARY IS OF FORM (KEY: post_word_pos);(VALUE: pre_word_pos)
#SO dict["verb"] == set(adverb, noun, ...) BUT NOT set(adjective, determiner, etc)
def possiblePartsSpeechPaths():
    #SO dict["verb"] == set(adverb, noun, ...) BUT NOT set(adjective, determiner, etc)
    pos_list = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS", "LS","MD","NN","NNS","NNP","NNPS", \
                "PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","TO","UH","VB","VBD","VBG","VBN","VBP", \
                "VBZ","WDT","WP","WP$","WRB"]
    dictTags = {}
    for tag in pos_list:
        s = set([])
        if("VB" in tag):
            s = set(["CC","RB","RBR","RBS","NN","NN","NNS","NNP","NNPS","MD","PRP"])
            sing_nouns = set(["NN","NNP"])
            plur_nouns = set(["NNS","NNPS"])
            if(tag in set(["VB","VBG","VBP","VBN"])):
                s.difference(sing_nouns)
            if(tag in set(["VBG","VBZ","VBN"])):
                s.difference(plur_nouns)
            if(tag in set(["VBG","VBN"])):
                s.union(set(["VB","VBD","VBP","VBZ"]))
        else:
            s=set(pos_list)
            if("IN"==tag):
                t = set(["IN","DT","CC"]) #maybe not CC
                s.difference(t)
            if("JJ" in tag):
                t = set(["NN","NNS","NNP","NNPS"])
                s.difference(t)
            if("TO"==tag):
                t = set(["DT","CC","IN"])
                s.difference(t)
            if("CC"==tag):
                t = set(["DT","JJ","JJR","JJS"])
                s.difference(t)
            if("NN" in tag):
                t = set(["NN","NNS","NNP","NNPS","PRP","CC"]) #maybe not CC
                s.difference(t)
            if("MD"==tag):
                t = set(["DT","VB","VBD","VBG","VBN","VBP","VBZ"])
                s.difference(t)
            if("PRP"==tag):
                t = set(["CC","JJ","JJR","JJS","NN","NNS","NNP","NNPS","DT"])
                s.difference(t)
            if("PRP$"==tag):
                t = set(["CC","DT","VB","VBD","VBG","VBN","VBP","VBZ","PRP"])
                s.difference(t)
            adv = set(["RB","RBR","RBS"])
            if(tag not in adv):
                s.remove(tag)
        dictTags[tag] = s
    return dictTags


class State:
    def __init__(self, tup):
        self.coord = tup #form: (line,stress)
        self.nexts = set()
        self.prevs = set()
        
        
def formLinkedTree(stress,dictMeters,corpus,fsaLine,dictWordTransitions,dictCorpusMeterGroups):
    test = 0
    if(stress==10):
        return "base_case"
    for meter in dictCorpusMeterGroups:
        if(isFitPattern(meter,stress)):
            new_stress = stress+len(meter)
            if(new_stress > 10):
                continue
            recursion = formLinkedTree(new_stress,dictMeters,corpus,fsaLine,dictWordTransitions,dictCorpusMeterGroups)
            if(recursion=="no_children"):
                continue
            if(recursion!="base_case"):
                fsaLine = recursion[0]
                dictWordTransitions = recursion[1]
            dictWordTransitions[(stress,new_stress)]=dictCorpusMeterGroups[meter]
            fsaLine[stress].nexts.add(new_stress)
            fsaLine[new_stress].prevs.add(stress)
            test += 1
    if(test==0):
        return "no_children"
    return fsaLine,dictWordTransitions


def simple_clean(string):
        string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'", "", string)
        string = re.sub(r"-", "", string)
        string = re.sub(r":", "", string)

        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'d", "ed", string) #for the old style
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"I\'ve", "I have", string)
        string = re.sub(r"\'ll", " will", string)

        string = re.sub(r"[0-9]+", "EOS", string) # EOS tag for numeric titling

        string = re.sub(r";", ",", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\.", " . ", string)

        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\s{2,}", " ", string)

        return string.strip().lower()
def partsOfSpeechFilter(word1,word2,dictPartSpeechTags,dictPossiblePartsSpeech):
            okay_tags = set(["RB","RBR","RBS"]) #THESE ARE THE ADVERBS
            try:
                tag1 = dictPartSpeechTags[word1]
                tag2 = dictPartSpeechTags[word2]
            except KeyError:
                return True
            #if(tag1==tag2 and tag1 not in okay_tags):
            #    return True
            if(tag1 not in dictPossiblePartsSpeech[tag2]):
                return True
            else:
                return False


def sampleLine(lst, cut):
    ''' samples from top "cut" lines, the distribution being the softmax of true sentence probabilities'''
    probs = list()
    C=min(cut, len(lst))
    for i in range(C):
        probs.append(np.exp(lst[i][0]))
    probs = np.exp(probs) / sum(np.exp(probs))
    index = np.random.choice(C,1,p=probs)[0]
    return lst[index][1]

def bool_init(word):
    postag=nltk.pos_tag(word)[0][1]
    if postag in set(['RB','RBR', 'RBS', 'VB', 'VBZ', 'LS','UH','PRP','MD']):
        return True
    return False
def init_word(vocabulary):
    init=random.choice(vocabulary)
    while bool_init([init]):
        init=random.choice(vocabulary)
    return init

def bool_last_word(sentence):
    last=nltk.pos_tag(sentence)
    last=last[-1][1]
    if last not in set(['EX','JJ','JJR','JJS','WDT','WP','WRB','PRP','RP','TO','DT']):
        return True
    return False

def last_word(vocabulary):
    init=random.choice(vocabulary)
    while (not bool_last_word([init])):
        init=random.choice(vocabulary)
    return init