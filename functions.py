import numpy as np
import tensorflow as tf
import argparse
import os
from six.moves import cPickle
import nltk
import time
import queue as Q
from operator import itemgetter
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
import random
from collections import defaultdict, Counter
import itertools
counts = defaultdict(Counter)

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

def createMeterGroups(corpus,dictMeters):#creates dict word to syllabus
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

def createPartSpeechTags(corpus):
    dictPartSpeechTags = {}
    for word in corpus:
        #if(word not in dictMeters):
        #    continue
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

        
        
def search_forward(model, vocab, prob_sequence, sequence, state, session, \
                temp, dictPartSpeechTags,dictPossiblePartsSpeech, breadth, wordPool, PartOfSpeachSet, TemplatePOS):
    def beamSearchOneLevel(model, vocab, prob_sequence, sequence, state, session, \
                    temp, dictPartSpeechTags,dictPossiblePartsSpeech, breadth, wordPool, PartOfSpeachSet, TemplatePOS):
        def decayRepeat(word,sequence, scale):
            safe_repeat_words = []
            #safe_repeat_words = set(["with,the,of,in,i"])
            score_adjust = 0
            decr = -scale
            for w in range(len(sequence)):
                if(word==sequence[w] and word not in safe_repeat_words):
                    score_adjust += decr
                decr += scale/10 #decreases penalty as the words keep getting further from the new word
            return score_adjust
        def partsOfSpeechFilter(word1,word2,dictPartSpeechTags,dictPossiblePartsSpeech):
            okay_tags = set(["RB","RBR","RBS"]) #THESE ARE THE ADVERBS
            try:
                tag1 = dictPartSpeechTags[word1]
            except KeyError:
                return True
            try:
                
                tag2 = dictPartSpeechTags[word2]
            except KeyError:
                return True
            #if(tag1==tag2 and tag1 not in okay_tags):
            #    return True
            if(tag1 not in dictPossiblePartsSpeech[tag2]):
                return True
            else:
                return False
        if(len(sequence)==len(TemplatePOS)):
            return("begin_line")
        ret = []
        scale = .02 #scale is the significant magnitude required to affect the score of bad/good things
        dist, state = model.compute_fx(session, vocab, prob_sequence, sequence, state, temp)
        #for pred_stress in list(fsaLine[post_stress].prevs):
        word_set = set([])
        #print (len(sequence))
        for word in list(set(PartOfSpeachSet[TemplatePOS[len(sequence)]])):
            #PREVENTS REPEAT ADJACENT WORDS OR PROBLEM-TAGGED WORDS
            if(word == sequence[-1]):
                continue
            if(partsOfSpeechFilter(sequence[-1],word,dictPartSpeechTags,dictPossiblePartsSpeech)):
                continue
            #FACTORS IN SCORE ADJUSTMENTS
            score_adjust = decayRepeat(word, sequence, 100*scale) #repeats
            score_adjust += scale*len(word)/50 #length word
            if(word in wordPool):
                score_adjust += scale
            #CALCULATES ACTUAL SCORE
            key = np.array([[vocab[word]]])
            new_prob = dist[key]
            score_tuple = (new_prob, state)
            score_tup = (score_tuple[0]+score_adjust,score_tuple[1]) #NOTE SHOULD SCORE_ADJUST BE ADDED HERE OR JUST IN THE ITEM LINE?
            item = (score_tup[0],(score_tup, sequence+[word]))
            if(item[0]==[[-float("inf")]]):
                continue
            ret+=[item]
        return ret
    masterPQ = Q.PriorityQueue()
    checkList = []
    checkSet = set([])
    score_tuple = (prob_sequence, state)
    first = (score_tuple[0],(score_tuple, sequence))
    masterPQ.put(first)#initial case
    set_explored = set([])
    while(not masterPQ.empty()):
        depthPQ = Q.PriorityQueue()
        while(not masterPQ.empty()):
            try:
                next_search = masterPQ.get()
            except:
                continue
            possible_branches = beamSearchOneLevel(model, vocab, next_search[1][0][0], next_search[1][1],\
                                next_search[1][0][1], session, temp,\
                                dictPartSpeechTags,dictPossiblePartsSpeech, breadth, wordPool, PartOfSpeachSet, TemplatePOS)
            if(possible_branches == "begin_line"):
                checkList+=[next_search]
                continue
            for branch in possible_branches:
                if(branch == []):
                    continue
                test = tuple(branch[1][1]) #need to make sure each phrase is being checked uniquely (want it to be checked once in possible branches then never again)
                if(test in set_explored):
                    continue
                set_explored.add(test)
                depthPQ.put(branch)
                try:
                    if(depthPQ.qsize()>breadth):
                        depthPQ.get()
                except:
                    pass
        masterPQ = depthPQ
    return checkList
    
    
def search_back(model, vocab, prob_sequence, sequence, state, session, \
                temp, dictPartSpeechTags,dictPossiblePartsSpeech, breadth, wordPool, PartOfSpeachSet, TemplatePOS):
    def beamSearchOneLevel(model, vocab, prob_sequence, sequence, state, session, \
                    temp, dictPartSpeechTags,dictPossiblePartsSpeech, breadth, wordPool, PartOfSpeachSet, TemplatePOS):
        def decayRepeat(word,sequence, scale):
            safe_repeat_words = []
            #safe_repeat_words = set(["with,the,of,in,i"])
            score_adjust = 0
            decr = -scale
            for w in range(len(sequence)):
                if(word==sequence[w] and word not in safe_repeat_words):
                    score_adjust += decr
                decr += scale/10 #decreases penalty as the words keep getting further from the new word
            return score_adjust
        def partsOfSpeechFilter(word1,word2,dictPartSpeechTags,dictPossiblePartsSpeech):
            okay_tags = set(["RB","RBR","RBS"]) #THESE ARE THE ADVERBS
            try:
                tag1 = dictPartSpeechTags[word1]
            except KeyError:
                return True
            tag2 = dictPartSpeechTags[word2]
            #if(tag1==tag2 and tag1 not in okay_tags):
            #    return True
            if(tag1 not in dictPossiblePartsSpeech[tag2]):
                return True
            else:
                return False
        if(len(sequence)==len(TemplatePOS)):
            return("begin_line")
        ret = []
        scale = .02 #scale is the significant magnitude required to affect the score of bad/good things
        dist, state = model.compute_fx(session, vocab, prob_sequence, sequence, state, temp)
        #for pred_stress in list(fsaLine[post_stress].prevs):
        word_set = set([])
        #print (len(sequence))
        for word in list(set(PartOfSpeachSet[TemplatePOS[-len(sequence)-1]])):
            #PREVENTS REPEAT ADJACENT WORDS OR PROBLEM-TAGGED WORDS
            if(word == sequence[0]):
                continue
            if(partsOfSpeechFilter(word,sequence[0],dictPartSpeechTags,dictPossiblePartsSpeech)):
                continue
            #FACTORS IN SCORE ADJUSTMENTS
            score_adjust = decayRepeat(word, sequence, 100*scale) #repeats
            score_adjust += scale*len(word)/50 #length word
            if(word in wordPool):
                score_adjust += scale
            #CALCULATES ACTUAL SCORE
            key = np.array([[vocab[word]]])
            new_prob = dist[key]
            score_tuple = (new_prob, state)
            score_tup = (score_tuple[0]+score_adjust,score_tuple[1]) #NOTE SHOULD SCORE_ADJUST BE ADDED HERE OR JUST IN THE ITEM LINE?
            item = (score_tup[0],(score_tup, [word]+sequence))
            if(item[0]==[[-float("inf")]]):
                continue
            ret+=[item]
        return ret
    masterPQ = Q.PriorityQueue()
    checkList = []
    checkSet = set([])
    score_tuple = (prob_sequence, state)
    first = (score_tuple[0],(score_tuple, sequence))
    masterPQ.put(first)#initial case
    set_explored = set([])
    while(not masterPQ.empty()):
        depthPQ = Q.PriorityQueue()
        while(not masterPQ.empty()):
            try:
                next_search = masterPQ.get()
            except:
                continue
            possible_branches = beamSearchOneLevel(model, vocab, next_search[1][0][0], next_search[1][1],\
                                next_search[1][0][1], session, temp,\
                                dictPartSpeechTags, dictPossiblePartsSpeech,breadth, wordPool, PartOfSpeachSet, TemplatePOS)
            if(possible_branches == "begin_line"):
                checkList+=[next_search]
                continue
            for branch in possible_branches:
                if(branch == []):
                    continue
                test = tuple(branch[1][1]) #need to make sure each phrase is being checked uniquely (want it to be checked once in possible branches then never again)
                if(test in set_explored):
                    continue
                set_explored.add(test)
                depthPQ.put(branch)
                try:
                    if(depthPQ.qsize()>breadth):
                        depthPQ.get()
                except:
                    pass
        masterPQ = depthPQ
    return checkList