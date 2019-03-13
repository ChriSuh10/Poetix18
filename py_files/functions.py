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

### Old collocation code
# from Collocation import Collocation
# counts = defaultdict(Counter)
#
# def grams(poets, window_size=5, filters=None):
#     grams = {}
#
#     # Iterate over all poets
#     for poet, filename in poets.items():
#         with open(filename, "r") as f:
#             for line in list(filter(lambda x: len(x) > 10, f)):
#                 line = line.strip()
#                 line = line.lower()
#                 # Uncomment to strip punctuation
#                 # line = re.sub(r'[^\w\s]', '', line)
#                 text = nltk.word_tokenize(line)
#                 template = [word[1] for word in nltk.pos_tag(text)]
#                 words = line.split()
#
#                 # Takes a guard function of parameters i and j
#                 def add_gram(guard, start, stop):
#                     # Sliding window
#                     for i in range(start, stop):
#                         for j in range(window_size):
#                             if guard(i, j):
#                                 continue
#
#                             gram = (words[i], words[i+j+1])
#                             gram_pos = tuple(map(lambda x: nltk.pos_tag(nltk.word_tokenize(x))[0][1], gram))
#
#                             if filters is None or gram_pos in filters:
#                                 coll = grams.get(gram, Collocation(gram))
#                                 coll.add(j+1)
#                                 grams[gram] = coll
#
#                 add_gram(lambda i, j: i+j+1 < 0, 0, len(words)-window_size)
#                 add_gram(lambda i, j: i+j+1 >= len(words) or i+j+1 < 0 or i < 0, len(words)-window_size, len(words))
#
#     return grams
#
# def get_collocation_dict(colls):
#     d = {}
#     for w1, w2 in colls.keys():
#         collocation = colls[(w1, w2)]
#         mean = collocation.mean()
#         std = collocation.standard_deviation()
#         if w1 not in d:
#             d[w1] = []
#         if w2 not in d:
#             d[w2] = []
#         d[w1].append([(w1, w2), mean, std])
#         d[w2].append([(w2, w1), -mean, std])
#     return d
#
# def build_get_collocation_dict(poets, window_size=5, filters=None):
#     return get_collocation_dict(grams(poets, window_size=window_size, filters=filters))

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
                "VBZ","WDT","WP","WP$","WRB",',']
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

# Modified to incorporate syllables
def search_back_meter(model, vocab, prob_sequence, sequence, state, session, \
                temp, dictPartSpeechTags, breadth, wordPool, PartOfSpeachSet, TemplatePOS,
                TemplateSyllables, dictSyllables):
    def beamSearchOneLevel(model, vocab, prob_sequence, sequence, state, session, \
                    temp, dictPartSpeechTags, breadth, wordPool, PartOfSpeachSet, TemplatePOS,
                    TemplateSyllables, dictSyllables):
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
                tag1 = dictPartSpeechTags[word1][0]
            except KeyError:
                return True
            tag2 = dictPartSpeechTags[word2][0]
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
        for word in list(set(PartOfSpeachSet[TemplatePOS[-len(sequence) - 1]])):
            #PREVENTS REPEAT ADJACENT WORDS OR PROBLEM-TAGGED WORDS
            if(word == sequence[0]):
                continue
            # if(partsOfSpeechFilter(word,sequence[0],dictPartSpeechTags,dictPossiblePartsSpeech)):
            #     continue
            # If cannot get syllables, don't use
            if (word not in dictSyllables):
                continue
            if len(dictSyllables[word][0]) != TemplateSyllables[-len(sequence) - 1]:
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
        if len(ret) < 1:
            print(sequence)
            print(TemplatePOS)
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
                                dictPartSpeechTags,breadth, wordPool, PartOfSpeachSet, TemplatePOS,\
                                TemplateSyllables, dictSyllables)
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



def get_templates():
    dataset={'NN': [
               (['CD', 'NN', 'VBD', 'TO', 'PRP$', 'NN'], ['one', 'morning', 'remarked', 'to', 'his', 'granny', ',']),
               (['DT', 'NN', 'IN', 'DT', 'JJ', 'NN'], ['the', 'point', 'of', 'an', 'undersized', 'pin']),
               (['PRP', 'VBD', 'IN', 'PRP$', 'NN'], ['he', 'bought', 'for', 'his', 'daughter']),
               (['TO', 'VB', 'JJ', 'NNS', 'IN', 'PRP$', 'NN'], ['to', 'balance', 'green', 'peas', 'on', 'her', 'fork']),
               (['TO', 'VB', 'DT', 'NN'], ['to', 'scare', 'off', 'the', 'critter', ',']),
               (['WP$', 'NN', 'VBD', 'RB', 'JJR', 'IN', 'NN'], ['whose', 'speed', 'was', 'much', 'faster', 'than', 'light', ',']),
               (['PRP', 'VBD', 'CD', 'NN'], ['she', "set", 'out', 'one', 'day']),
               (['IN', 'DT', 'JJ', 'NN'], ['in', 'a', 'relative', 'way']),
               (['CC', 'VBN', 'IN', 'DT', 'JJ', 'NN'], ['and', 'returned', 'on', 'the', 'previous', 'nighty']),
               (['RB', 'VBD', 'IN', 'DT', 'NN', 'IN', 'DT', 'NN'], ['once', 'fished', 'from', 'the', 'edge', 'of', 'a', 'fissure']),
                (['DT', 'NN', ',', 'IN', 'JJ', 'NN'], ['a', 'major', ',', 'with', 'wonderful', 'force']),
               (['IN', 'PRP$', 'NN', 'VBD', 'PRP$', 'NN'], ['that', 'his', 'back', 'touched', 'his', 'chest']),
               (['PRP', 'VBD', 'IN', 'DT', 'NN', 'IN','DT', 'NN'], ['she', 'slipped', 'on', 'a', 'peel', 'of', 'banana']),
                (['IN', 'PRP', 'VBD', 'IN', 'PRP', 'NN'], ['as', 'she', 'lay', 'on', 'her', 'side']),
               (['IN', 'VBG', 'VBD', 'VBN', 'IN', 'DT', 'NN'], ['while', 'walking', 'was', 'caught', 'in', 'the', 'rain'])],

         'VBN':[(['IN', 'DT', 'JJ', 'NN', 'VBD', 'VBN'], ['in', 'a', 'funeral', 'procession', 'was', 'spied']),
                (['CC', 'DT', 'NN', 'MD', 'VB', 'VBN'], ['but', 'no', 'horse', 'could', 'be', 'found']),
                (['CC', 'RB', 'PRP', 'MD', 'VB', 'VBN'], ['and', 'sideways', 'he', "couldn't", 'be', 'seen'])
               ],
         'VB':[(['VBD', 'TO', 'VB', 'CD', 'JJ', 'NNS', 'TO', 'VB'], ['tried', 'to', 'teach', 'two', 'young', 'tooters', 'to', 'toot'])],
         'VBD':[(['PRP', 'VBD', 'CC', 'VBD'], ['he', 'giggled', 'and', 'said', ',']),
               (['PRP', 'VBD', 'IN', 'NN', 'CC', 'VBD'], ['she', 'sat', 'up', 'in', 'bed', 'and', 'meowed']),
                (['CC', 'DT', 'NN', 'PRP', 'VBD'], ['but', 'the', 'copy', 'he', 'wrote']),
                (['RBR', 'NNS', 'PRP', 'VBD'], ['more', 'stars', 'she', 'espied']),
                (['PRP', 'VBD', ',', 'RB', 'VBD'], ['she', 'ran', ',', 'almost', 'flew']),
                (['CC', 'NN', 'VBZ', 'WDT', 'NN', 'PRP', 'VBD'], ['and', 'no', 'one', 'knows', 'which', 'way', 'she', 'went']),

               ],

        'VBZ':[(['CC', 'DT', 'NN', 'IN', 'PRP', 'VBZ'], ['but', 'the', 'welt', 'that', 'he', 'raises']),
              (['CC', 'DT', 'VBZ', 'WRB', 'DT', 'NN', 'VBZ'], ['and', 'that', 'is', 'where', 'the', 'rub', 'comes', 'in']),
              ],
        'PRP':[(['DT', 'NN', 'WP', 'VBD', 'PRP'], ['a', 'tutor', 'who', 'taught', 'her']),
              (['RB', 'DT', 'JJ', 'NN', 'VBD', 'PRP'], ['soon', 'a', 'happy', 'thought', 'hit', 'her'])],

        'JJ':[
             (['DT', 'NN', ',', 'RB', 'JJ'],['a', 'canner', ',', 'exceedingly', 'canny', ',']),
             (['DT', 'NN', 'WDT', 'VBZ', 'DT', 'JJ'], ['the', 'bug', 'that', 'is', 'no', 'big']),
              (['PDT', 'DT', 'NNS', 'VBD', 'JJ'], ['all', 'the', 'flowers', 'looked', 'round']),
              (['PRP', 'VBD', 'RB', 'RB', 'JJ'], ['she', 'grew', 'so', 'abnormally', 'lean']),
              (['CC', 'JJ', ',', 'CC', 'VBD'], ['and', 'flat', ',', 'and', 'compressed']),
              (['CC', 'PRP', 'VBD', 'NN', 'RB', 'JJ'], ['and', 'she', 'reached', 'home', 'exceedingly', 'plain'])
             ],

        'RB':[(['PRP$', 'NN', 'VBD', 'RB'], ['her', 'complexion', 'did', 'too'])]}
    third_line={'NN':[(['PRP', 'MD', 'VB', 'TO', 'DT', 'NN'], 'He would go to a party'),
    (['VBN', 'IN', 'DT', 'NN'], ['collapsed', 'from', 'the', 'strain']),
               (['PRP', 'VBD', 'CD', 'NN'], ['she', "set", 'out', 'one', 'day']),
               (['IN', 'DT', 'JJ', 'NN'], ['in', 'a', 'relative', 'way']),
                (['IN', 'PRP$', 'NN', 'VBD', 'PRP$', 'NN'], ['that', 'his', 'back', 'touched', 'his', 'chest']),
                  (['IN', 'PRP', 'VBD', 'IN', 'PRP', 'NN'], ['as', 'she', 'lay', 'on', 'her', 'side']),
                 (['PRP', 'VBD', 'PRP', 'NN'], ['she', 'followed', 'her', 'nose'])
],
            'VBN':[(['CC', 'DT', 'NN', 'MD', 'VB', 'VBN'], ['but', 'no', 'horse', 'could', 'be', 'found']),

                  ],

            'JJ':[(['CC', 'VB', 'RB', 'IN', 'JJ'], 'And eat just as hearty'),
                  (['PDT', 'DT', 'NNS', 'VBD', 'NN'], ['all', 'the', 'flowers', 'looked', 'round']),
                  (['CC', 'JJ', ',', 'CC', 'VBD'], ['and', 'flat', ',', 'and', 'compressed']),


            ],

           'VBD':[(['PRP', 'VBD', 'CC', 'VBD'], ['he', 'giggled', 'and', 'said', ',']),
                 (['CC', 'DT', 'NN', 'PRP', 'VBD'], ['but', 'the', 'copy', 'he', 'wrote']),
                  (['RBR', 'NNS', 'PRP', 'VBD'], ['more', 'stars', 'she', 'espied']),
                 (['PRP', 'VBD', ',', 'RB', 'VBD'], ['she', 'ran', ',', 'almost', 'flew']),
                 (['CD', 'NN', ',', 'PRP', 'VBD'], ['one', 'day', ',', 'i', 'supposed'])],
           'RB':[(['PRP$', 'NN', 'VBD', 'RB'], ['her', 'complexion', 'did', 'too'])],
           }
    second_line={'JJ':[
                  (['PRP', 'VBD', 'RB', 'RB', 'JJ'], ['she', 'grew', 'so', 'abnormally', 'lean']),
                  (['WP$', 'NN', 'VBD', 'RB', 'JJ'], ['whose', 'nose', 'was', 'awfully', 'bent']),
                  (['PRP', 'VBD', 'RB', 'RB', 'JJ'], ['who', 'was', 'so', 'excessively', 'thin'])],
            'NN':[['PRP', 'MD', 'VB', 'TO', 'DT', 'NN'], (['PRP', 'VBD', 'IN', 'PRP$', 'NN'], ['he', 'bought', 'for', 'his', 'daughter']),
                 (['WP$', 'NN', 'VBD', 'RB', 'JJR', 'IN', 'NN'], ['whose', 'speed', 'was', 'much', 'faster', 'than', 'light', ',']),
                 (['PRP', 'VBD', 'CD', 'NN'], ['she', 'set', 'out', 'one', 'day', ',']),
                  (['PRP', 'VBD', 'IN', 'DT', 'NN', 'IN', 'NN'], ['she', 'slipped', 'on', 'a', 'peel', 'of', 'banana']),
                 ],
            'VBN':[(['IN', 'DT', 'JJ', 'NN', 'VBD', 'VBN'], ['in', 'a', 'funeral', 'procession', 'was', 'spied'])],
            'VBD':[(['PRP', 'VBD', 'CC', 'VBD'], ['he', 'giggled', 'and', 'said', ','])]}
    return dataset, second_line, third_line


def get_templates_new():
    dataset={'NN': [(['DT', 'JJ', 'JJ', 'NN', 'VBN', 'NN', 'NN'], ['a', 'certain', 'young', 'fellow', 'named', 'bee', 'bee']), (['VBN', 'TO', 'VB', 'DT', 'NN', 'VBN', 'NN'], ['wished', 'to', 'wed', 'a', 'woman', 'named', 'phoebe']), (['WP', 'DT', 'JJ', 'NN'], ['what', 'the', 'clerical', 'fee']), (['VB', 'IN', 'JJ', 'VB', 'JJ', 'NN', 'NN'], ['be', 'before', 'phoebe', 'be', 'phoebe', 'bee', 'bee']), (['PRP', 'MD', 'VB', 'TO', 'DT', 'NN'], ['he', 'would', 'go', 'to', 'a', 'party']), (['VBD', 'PRP$', 'NN', ',', 'PRP$', 'NN'], ['said', 'her', 'doctor', ',', 'its', 'plain']), (['VBN', 'CD', 'NNS', 'IN', 'PRP$', 'NN'], ['interrupted', 'two', 'girls', 'with', 'their', 'knittin']), (['RB', 'VBN', 'PRP', ',', 'RB', 'WRB', 'NN', 'NN'], ['just', 'painted', 'it', ',', 'right', 'where', 'youre', 'sittin']), (['IN', 'DT', 'NN', 'IN', 'DT', 'JJ', 'NN'], ['than', 'the', 'point', 'of', 'an', 'undersized', 'pin']), (['DT', 'JJ', 'JJ', 'NN', 'IN', 'NN'], ['a', 'silly', 'young', 'man', 'from', 'clyde']), (['JJ', 'VBP', 'VBP', 'NN', 'RB', 'VBD', 'IN', 'DT', 'NN'], ['i', 'dont', 'know', 'i', 'just', 'came', 'for', 'the', 'ride']), (['WP$', 'NN', 'VBD', 'DT', 'NN', 'IN', 'NN'], ['whose', 'pa', 'made', 'a', 'fortune', 'in', 'pork']), (['PRP', 'VBD', 'IN', 'PRP$', 'NN'], ['he', 'bought', 'for', 'his', 'daughter']), (['TO', 'VB', 'JJ', 'NNS', 'IN', 'PRP', 'NN'], ['to', 'balance', 'green', 'peas', 'on', 'her', 'fork']), (['DT', 'NN', 'IN', 'PRP$', 'NN', 'VBD', 'JJ', 'NN'], ['a', 'mouse', 'in', 'her', 'room', 'woke', 'miss', 'dowd']), (['CC', 'VBN', 'IN', 'DT', 'JJ', 'NN'], ['and', 'returned', 'on', 'the', 'previous', 'night']), (['WP', 'IN', 'NNS', 'VBD', 'RB', 'NN', 'IN', 'NN'], ['who', 'on', 'apples', 'was', 'quite', 'fond', 'of', 'feedin']), (['CC', 'RB', 'DT', 'VBN', 'IN', 'NN'], ['and', 'then', 'both', 'skedaddled', 'from', 'eden']), (['NNS', 'CD', 'NN', 'NN', 'VBP', 'NN'], ['theres', 'one', 'thing', 'i', 'cannot', 'determine']), (['NNS', 'DT', 'NN', 'IN', 'NN'], ['shes', 'a', 'person', 'of', 'note']), (['DT', 'NN', 'JJ', 'NN', 'VBD', 'NN'], ['a', 'canny', 'young', 'fisher', 'named', 'fisher']), (['RB', 'VBN', 'IN', 'DT', 'NN', 'IN', 'DT', 'NN'], ['once', 'fished', 'from', 'the', 'edge', 'of', 'a', 'fissure']), (['DT', 'NN', 'IN', 'DT', 'NN'], ['a', 'fish', 'with', 'a', 'grin']), (['RB', 'VBZ', 'VBG', 'DT', 'NN', 'IN', 'NN'], ['now', 'theyre', 'fishing', 'the', 'fissure', 'for', 'fisher']), (['MD', 'VB', 'NN', 'IN', 'DT', 'JJ', 'NN'], ['could', 'make', 'copy', 'from', 'any', 'old', 'thing']), (['IN', 'DT', 'CD', 'NN', 'NN'], ['of', 'a', 'five', 'dollar', 'note']), (['VBD', 'RB', 'JJ', 'PRP', 'VBZ', 'RB', 'IN', 'VBG', 'NN'], ['was', 'so', 'good', 'he', 'is', 'now', 'in', 'sing', 'sing']), (['CC', 'PRP', 'VBD', 'IN', 'DT', 'NN', 'IN', 'DT', 'NN'], ['and', 'they', 'flew', 'through', 'a', 'flaw', 'in', 'the', 'flue']), (['DT', 'NN', 'WP', 'VBD', 'DT', 'NN'], ['a', 'tutor', 'who', 'tooted', 'a', 'flute']), (['VBN', 'RP', 'IN', 'NN', 'NN', 'IN', 'DT', 'NN'], ['called', 'out', 'in', 'hyde', 'park', 'for', 'a', 'horse']), (['RB', 'PRP', 'RB', 'VB', ',', 'IN', 'NN'], ['so', 'he', 'just', 'rhododendron', ',', 'of', 'course']), (['WP', 'VBD', 'IN', 'DT', 'NN', 'IN', 'NN'], ['who', 'slipped', 'on', 'a', 'peel', 'of', 'banana']), (['IN', 'EX', 'VBP', 'IN', 'DT', 'NN', 'VBD', 'NN'], ['than', 'there', 'are', 'in', 'the', 'star', 'spangled', 'banner']), (['CC', 'VBD', 'DT', 'NN', 'IN', 'TO', 'VB', 'NN'], ['and', 'invented', 'a', 'scheme', 'for', 'to', 'scare', 'em']), (['PRP', 'VBD', 'PRP', 'DT', 'NN'], ['he', 'caught', 'him', 'a', 'mouse']), (['WDT', 'PRP', 'VBD', 'IN', 'DT', 'NN'], ['which', 'he', 'loosed', 'in', 'the', 'house']), (['IN', 'DT', 'NN', 'VBZ', 'VBN', 'NN', 'NN', 'NN'], ['\\(', 'the', 'confusion', 'is', 'called', 'harem', 'scarem', '\\)']), (['DT', 'JJ', 'JJ', 'NN', 'VBN', 'NN'], ['a', 'nifty', 'young', 'flapper', 'named', 'jane']), (['IN', 'VBG', 'VBD', 'VBN', 'IN', 'DT', 'NN'], ['while', 'walking', 'was', 'caught', 'in', 'the', 'rain']), (['IN', 'NN', 'DT', 'NN'], ['of', 'course', 'the', 'expense']), (['CC', 'PRP', 'VBZ', 'VBN', 'IN', 'IN', 'PRP$', 'NN'], ['but', 'it', 'doesnt', 'come', 'out', 'of', 'my', 'purse']), (['NNS', 'DT', 'NN', 'IN', 'JJ', 'NN', ',', 'VBD', 'JJ', 'NN'], ['theres', 'a', 'train', 'at', 'eos', 'eos', ',', 'said', 'miss', 'jenny']), (['PRP', 'VBD', 'PRP', 'NN'], ['she', 'followed', 'her', 'nose']), (['TO', 'VB', 'NN'], ['to', 'drink', 'lemonade']), (['WP', 'VBD', 'DT', 'JJ', 'NN', 'DT', 'NN'], ['who', 'read', 'a', 'love', 'story', 'each', 'day']), (['JJ', 'NN', 'VBP', 'NN', 'VBD', 'DT', 'NN'], ['i', 'didnt', 'think', 'life', 'was', 'this', 'way']), (['IN', 'PRP', 'VBD', 'RP', 'CD', 'NNS', 'IN', 'PRP$', 'NN'], ['as', 'she', 'let', 'out', 'three', 'tucks', 'in', 'her', 'tunic']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'IN', 'NN'], ['there', 'was', 'an', 'old', 'man', 'of', 'nantucket']), (['WP', 'VBD', 'DT', 'PRP$', 'NN', 'IN', 'DT', 'NN'], ['who', 'kept', 'all', 'his', 'cash', 'in', 'a', 'bucket']), (['CC', 'PRP$', 'NN', ',', 'VBN', 'NN'], ['but', 'his', 'daughter', ',', 'named', 'nan']), (['VB', 'RP', 'IN', 'DT', 'NN'], ['ran', 'away', 'with', 'a', 'man']), (['CC', 'RB', 'RB', 'IN', 'DT', 'NN', ',', 'NN'], ['and', 'as', 'far', 'as', 'the', 'bucket', ',', 'nantucket']), (['WP', 'VBD', 'IN', 'PRP', 'VBD', 'IN', 'DT', 'NN'], ['who', 'smiled', 'as', 'she', 'rode', 'on', 'a', 'tiger']), (['PRP', 'VBD', 'RB', 'IN', 'DT', 'NN'], ['they', 'came', 'back', 'from', 'the', 'ride']), (['CC', 'DT', 'NN', 'IN', 'DT', 'NN', 'IN', 'DT', 'NN'], ['and', 'the', 'smile', 'on', 'the', 'face', 'of', 'the', 'tiger']), (['DT', 'NN', ',', 'JJ', 'NN'], ['a', 'dark', ',', 'disagreeable', 'fellow']), (['RB', 'PRP', 'VBD', 'PRP$', 'JJ', 'NN'], ['then', 'he', 'took', 'his', 'own', 'life']), (['PRP$', 'DT', 'NNS', 'IN', 'NN', 'IN', 'JJ', 'NN'], ['its', 'the', 'people', 'in', 'front', 'that', 'i', 'jar']), (['NN', 'NN', 'IN', 'DT', 'JJ', 'JJ', 'NN'], ['dont', 'proceed', 'in', 'the', 'old', 'fashioned', 'way']), (['WP', 'VBD', 'DT', 'NN', 'IN', 'NN'], ['who', 'hadnt', 'an', 'atom', 'of', 'fear']), (['PRP', 'VBD', 'DT', 'NN'], ['he', 'indulged', 'a', 'desire']), (['TO', 'VB', 'DT', 'JJ', 'NN'], ['to', 'touch', 'a', 'live', 'wire']), (['RB', 'JJS', 'DT', 'JJ', 'NN', 'MD', 'VB', 'RB', '.', 'NN'], ['\\(', 'most', 'any', 'last', 'line', 'will', 'do', 'here', '!', '\\)']), (['DT', 'JJ', 'JJ', 'NN', 'IN', 'DT', 'NN'], ['a', 'cheerful', 'old', 'bear', 'at', 'the', 'zoo']), (['IN', 'DT', 'JJ', 'JJ', 'NN', 'VBD', 'NN'], ['through', 'that', 'silly', 'scent', 'willie', 'sent', 'millicent']), (['WP', 'VBD', 'RP', 'PRP$', 'NNS', 'IN', 'DT', 'NN'], ['who', 'sent', 'out', 'his', 'cards', 'for', 'a', 'party']), (['PRP', 'VBD', 'NN', 'IN', 'NN'], ['it', 'filled', 'galileo', 'with', 'mirth']), (['RB', 'NN', 'VBD', 'IN', 'JJ', 'NN'], ['then', 'newton', 'announced', 'in', 'due', 'course']), (['PRP$', 'JJ', 'NN', 'IN', 'JJ', 'NN'], ['his', 'own', 'law', 'of', 'gravitys', 'force']), (['IN', 'DT', 'JJ', 'NN'], ['as', 'the', 'inverted', 'square']), (['IN', 'DT', 'NN', 'IN', 'NN', 'TO', 'NN'], ['of', 'the', 'distance', 'from', 'object', 'to', 'source']), (['CC', 'RB', ',', 'VBZ', 'NN'], ['but', 'remarkably', ',', 'einsteins', 'equation']), (['VBZ', 'TO', 'VB', 'NN'], ['succeeds', 'to', 'describe', 'gravitation']), (['IN', 'DT', 'NNS', 'JJ', 'NN'], ['as', 'the', 'planets', 'unique', 'motivation']), (['DT', 'JJ', 'NN', 'IN', 'NN', 'NN'], ['an', 'elderly', 'bride', 'of', 'port', 'jervis']), (['VBD', 'RB', 'JJ', 'NN'], ['was', 'quite', 'understandable', 'nervis']), (['NN', 'VBG', 'PRP', 'IN', 'DT', 'NN'], ['kept', 'insuring', 'her', 'during', 'the', 'service']), (['VB', 'PRP', 'VB', ',', 'VBD', 'DT', 'NN'], ['let', 'us', 'fly', ',', 'said', 'the', 'flea']), (['RB', 'PRP', 'VBP', 'IN', 'DT', 'NN', 'IN', 'DT', 'NN'], ['so', 'they', 'flew', 'through', 'a', 'flaw', 'in', 'the', 'flue']), (['VBD', 'PRP', 'IN', 'WP', 'DT', 'NN'], ['switched', 'it', 'on', 'what', 'a', 'din']), (['CC', 'RB', 'VBZ', 'DT', 'NN', 'NN'], ['and', 'now', 'hes', 'a', 'college', 'professor']), (['CC', 'VBD', 'NN', 'IN', 'PRP$', 'NN'], ['and', 'made', 'cider', 'inside', 'her', 'inside']), (['DT', 'NN', 'NN', 'VBN', 'NN'], ['a', 'crossword', 'compiler', 'named', 'moss']), (['WP', 'VBD', 'PRP', 'RB', 'IN', 'DT', 'NN'], ['who', 'found', 'himself', 'quite', 'at', 'a', 'loss']), (['VBD', ',', 'JJ', 'VBP', 'DT', 'NN'], ['said', ',', 'i', 'havent', 'a', 'clue']), (['NN', 'VBD', ',', 'PRP', 'VBZ', 'DT', 'JJ', 'NN', 'IN', 'NN'], ['bo', 'said', ',', 'it', 'is', 'a', 'polar', 'bear', 'in', 'snow']), (['NN', 'VBG', 'NNS', 'IN', 'DT', 'NN'], ['im', 'papering', 'walls', 'in', 'the', 'loo']), (['CC', 'RB', 'JJ', 'NN', 'VBD', 'DT', 'NN'], ['and', 'quite', 'frankly', 'i', 'havent', 'a', 'clue']), (['CC', 'VB', 'VBN', 'TO', 'DT', 'NN', 'IN', 'NN'], ['and', 'im', 'stuck', 'to', 'the', 'toilet', 'with', 'glue']), (['DT', 'NNS', 'IN', 'JJ', 'NN', 'NN'], ['the', 'shoes', 'of', 'old', 'eskimo', 'joe']), (['VBD', 'RB', 'IN', 'PRP', 'VBD', 'IN', 'DT', 'NN'], ['fell', 'apart', 'as', 'he', 'walked', 'in', 'the', 'snow']), (['WP', 'VBD', 'IN', 'JJ', 'NN'], ['who', 'lived', 'on', 'distilled', 'kerosene']), (['CC', 'PRP', 'VBD', 'NN'], ['but', 'she', 'started', 'absorbin']), (['DT', 'NN', 'IN', 'PRP$', 'NN'], ['the', 'cause', 'of', 'his', 'sorrow']), (['VBD', 'NN'], ['was', 'paradichloro']), (['NN'], ['triphenyldichloroethane']), (['VBZ', 'PRP', 'PRP', 'CC', 'DT', 'NN', 'IN', 'NN'], ['is', 'it', 'me', 'or', 'the', 'nature', 'of', 'money']), (['CC', 'WRB', 'NNS', 'VBP', 'NN'], ['but', 'when', 'i', 'have', 'dough']), (['CC', 'VBZ', 'IN', 'IN', 'PRP$', 'NNS', 'IN', 'NN'], ['and', 'seeps', 'out', 'of', 'my', 'pockets', 'like', 'honey']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'IN', 'DT', 'NN'], ['there', 'was', 'an', 'old', 'man', 'with', 'a', 'beard']), (['CD', 'NNS', 'CC', 'DT', 'NN'], ['four', 'larks', 'and', 'a', 'wren']), (['IN', 'PRP$', 'NN', 'IN', 'NN'], ['in', 'my', 'pocket', 'for', 'cash']), (['IN', 'PRP$', 'RB', 'JJ', 'NN', 'IN', 'DT', 'NN'], ['on', 'your', 'very', 'first', 'date', 'as', 'a', 'teen']), (['WP', 'VBD', 'VBN', 'IN', 'DT', 'NN', 'NN'], ['what', 'was', 'shown', 'on', 'the', 'cinema', 'screen']), (['DT', 'JJ', 'NN', 'IN', 'NN'], ['the', 'incredible', 'wizard', 'of', 'oz']), (['VBN', 'IN', 'PRP$', 'NN', 'NN'], ['retired', 'from', 'his', 'business', 'becoz']), (['NNS', 'CD', 'NN', 'NN', 'VBP', 'NN'], ['theres', 'one', 'thing', 'i', 'cannot', 'determine']), (['NNS', 'DT', 'NN', 'IN', 'NN'], ['shes', 'a', 'person', 'of', 'note']), (['WRB', 'NN', 'VBP', 'PRP', ',', 'VB', 'VBN', 'RB', 'NN'], ['when', 'i', 'wear', 'it', ',', 'im', 'called', 'only', 'vermin']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'IN', 'NN'], ['there', 'was', 'an', 'old', 'person', 'of', 'fratton']), (['IN', 'JJ', 'NNS', 'IN', 'PRP$', 'NN'], ['with', 'noxious', 'smells', 'in', 'my', 'nose']), (['RB', ',', 'VBP', 'NN'], ['amazingly', ',', 'antelope', 'stew']), (['IN', 'DT', 'NN', 'IN', 'NN'], ['than', 'a', 'goulash', 'of', 'rat']), (['CC', 'JJ', 'NN'], ['or', 'hungarian', 'cat']), (['IN', 'JJ', 'VBP', 'VBP', 'DT', 'NN'], ['if', 'i', 'didnt', 'curb', 'the', 'sound']), (['NNS', 'VBD', 'PRP', ',', 'VB', 'IN', 'IN', 'PRP$', 'NN'], ['i', 'told', 'him', ',', 'get', 'out', 'of', 'my', 'place']), (['NN', 'DT', 'NN', 'VBD', 'NN'], ['youre', 'an', 'utter', 'uncultured', 'disgrace']), (['NN', 'DT', 'NN', 'NN'], ['youre', 'a', 'simpleton', 'loon']), (['RB', 'PRP', 'VBD', 'PRP', 'NN', 'IN', 'DT', 'NN'], ['then', 'he', 'walloped', 'me', 'square', 'in', 'the', 'face']), (['DT', 'JJ', 'NN', 'NN', 'IN', 'NN'], ['a', 'young', 'gourmet', 'dining', 'at', 'crewe']), (['VBD', 'DT', 'RB', 'JJ', 'NN', 'IN', 'PRP$', 'NN'], ['found', 'a', 'rather', 'large', 'mouse', 'in', 'his', 'stew']), (['VBD', 'DT', 'NN', ',', 'NN', 'NN'], ['said', 'the', 'waiter', ',', 'dont', 'shout']), (['PRP', 'VBD', 'IN', 'DT', 'NN'], ['it', 'said', 'on', 'the', 'door']), (['NN', 'NN', 'IN', 'DT', 'NN'], ['dont', 'spit', 'on', 'the', 'floor']), (['RB', 'PRP', 'VBD', 'RB', 'CC', 'VB', 'IN', 'DT', 'NN'], ['so', 'he', 'jumped', 'up', 'and', 'spat', 'on', 'the', 'ceiling']), (['PRP', 'RB', 'RB', 'IN', 'DT', 'NN'], ['it', 'right', 'there', 'on', 'the', 'spot']), (['IN', 'PRP', 'VBD', 'TO', 'VB', ',', 'VB', 'DT', 'NN'], ['as', 'it', 'tried', 'to', 'explain', ',', 'im', 'a', 'spi']), (['DT', 'NN', 'IN', 'PRP$', 'NN'], ['every', 'night', 'in', 'his', 'shed']), (['DT', 'NN', 'NN', 'VBN', 'NN'], ['a', 'motor', 'mechanic', 'named', 'fox']), (['IN', 'PRP$', 'NNS', 'CC', 'PRP$', 'NN'], ['in', 'his', 'boots', 'and', 'his', 'vest']), (['IN', 'PRP$', 'NN', 'CC', 'NN', 'IN', 'DT', 'NN'], ['with', 'his', 'spanner', 'and', 'jack', 'in', 'the', 'box']), (['DT', 'NNS', 'NN', 'IN', 'NN'], ['a', 'cheesemongers', 'shop', 'in', 'paree']), (['VBN', 'TO', 'DT', 'NN'], ['collapsed', 'to', 'the', 'ground']), (['IN', 'DT', 'JJ', 'NN'], ['with', 'a', 'thunderous', 'sound']), (['VB', 'DT', 'RB', ',', 'JJ', 'NN'], ['cooed', 'the', 'shapely', ',', 'urbane', 'debutante']), (['NNS', 'VBD', 'RP', 'TO', 'NN'], ['didnt', 'rush', 'off', 'to', 'town']), (['NN', 'VBD', 'WRB', 'NN', 'VBP', 'IN', 'NN'], ['i', 'relaxed', 'when', 'i', 'eos', 'across', 'aunt']), (['DT', 'JJ', 'NN', 'VBN', 'NN'], ['an', 'elderly', 'man', 'called', 'keith']), (['RB', 'RB', ',', 'CC', 'VBD', 'VBN', 'NN'], ['sat', 'down', ',', 'and', 'was', 'bitten', 'beneath']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'VBN', 'NN'], ['there', 'was', 'a', 'young', 'lady', 'named', 'harris']), (['VBN', 'RP', 'TO', 'VB', 'NN', 'IN', 'NN'], ['turned', 'out', 'to', 'be', 'plaster', 'of', 'paris']), (['TO', 'VB', 'VBG', 'DT', 'NN', 'DT', 'NN', 'NN'], ['to', 'start', 'giving', 'this', 'house', 'a', 'spring', 'clean']), (['UH', ',', 'JJ', 'VBP', 'PRP', 'NN'], ['yes', ',', 'ill', 'do', 'it', 'today']), (['IN', 'PRP$', 'NNS', 'CC', 'PRP$', 'NN'], ['in', 'my', 'legs', 'and', 'my', 'bum']), (['MD', 'VB', 'DT', 'NN', 'IN', 'NN'], ['may', 'evolve', 'a', 'professor', 'at', 'yale']), (['DT', 'PRP', 'VBZ', 'VBG', 'NN'], ['a', 'he', 'melon', 'suffering', 'droop']), (['VBD', 'DT', 'PRP', 'NN', 'NN', 'IN', 'DT', 'NN'], ['spied', 'a', 'she', 'melon', 'round', 'as', 'a', 'hoop']), (['CC', 'PRP', 'VBD', 'CC', 'PRP', 'VBD', ',', 'NN'], ['but', 'she', 'sighed', 'and', 'she', 'said', ',', 'canteloupe']), (['WP', 'VBD', 'DT', 'JJ', 'NN', 'IN', 'PRP$', 'NN'], ['who', 'had', 'a', 'large', 'wart', 'on', 'her', 'nose']), (['CC', 'VBG', 'IN', 'NN', 'CC', 'NN'], ['and', 'inspiring', 'in', 'meter', 'and', 'rhyme']), (['IN', 'JJ', 'NN'], ['with', 'intelligent', 'thought']), (['CC', 'TO', 'VB', 'PRP', 'JJ', 'NNS', 'IN', 'NN'], ['and', 'to', 'write', 'it', 'used', 'acres', 'of', 'time']), (['EX', 'RB', 'VBD', 'DT', 'NN', 'IN', 'DT', 'NN'], ['there', 'once', 'was', 'a', 'fly', 'on', 'the', 'wall']), (['DT', 'JJ', 'NN', 'IN', 'DT', 'JJ', 'NN'], ['a', 'long', 'time', 'ago', 'an', 'old', 'squire']), (['VBD', 'DT', 'RB', 'JJ', 'NN', 'IN', 'DT', 'NN'], ['met', 'a', 'pretty', 'young', 'lass', 'in', 'a', 'choir']), (['CC', 'PRP', 'VBD', 'PRP', ',', 'DT', 'NN'], ['but', 'she', 'told', 'him', ',', 'no', 'chance']), (['IN', 'JJ', 'VBP', 'IN', 'NN', 'NNS', 'VBP', ',', 'NN'], ['for', 'i', 'fear', 'that', 'im', 'handels', 'miss', ',', 'sire']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'VBN', 'NN'], ['there', 'was', 'a', 'young', 'fellow', 'called', 'binn']), (['TO', 'VB', 'NN'], ['to', 'drink', 'lemonade']), (['NNS', 'VBP', 'DT', 'JJ', 'NN', 'IN', 'PRP$', 'NN'], ['i', 'need', 'a', 'front', 'door', 'for', 'my', 'hall']), (['DT', 'JJ', 'VBG', 'NN', 'IN', 'NN'], ['an', 'odd', 'looking', 'guy', 'from', 'beruit']), (['VBN', 'RP', 'NNS', 'IN', 'DT', 'JJ', 'JJ', 'NN'], ['held', 'up', 'banks', 'in', 'a', 'bright', 'yellow', 'suit']), (['PRP', 'MD', 'VB', 'DT', 'NN'], ['he', 'would', 'wave', 'a', 'cigar']), (['CC', 'NN', ',', 'NN', 'RB', ',', 'CC', 'RB', 'JJ', 'NN'], ['and', 'shout', ',', 'freeze', 'there', ',', 'or', 'else', 'ill', 'cheroot'])], 'VB': [(['CC', ',', 'PRP', 'VBD', ',', 'JJ', 'MD', 'VB'], ['but', ',', 'he', 'said', ',', 'i', 'must', 'see']), (['DT', 'NN', 'NN', 'RB', 'VB'], ['that', 'park', 'bench', 'well', 'i']), (['VBN', 'TO', 'VB', 'CD', 'JJ', 'NNS', 'TO', 'VB'], ['tried', 'to', 'teach', 'two', 'young', 'tooters', 'to', 'toot']), (['UH', 'WRB', 'JJ', '.', 'DT', 'NNS', 'MD', 'VB'], ['oh', 'how', 'modern', '!', 'the', 'critics', 'will', 'say']), (['MD', 'RB', 'VB', 'NN', 'TO', 'VB'], ['could', 'always', 'find', 'something', 'to', 'do']), (['VBD', 'RB', 'VBG', 'TO', 'VB'], ['was', 'highly', 'displeasing', 'to', 'millicent']), (['TO', 'VB', 'PRP$', 'CD', 'NNS', 'VBP', 'TO', 'VB'], ['to', 'watch', 'his', 'two', 'rocks', 'fall', 'to', 'earth']), (['CC', 'PRP$', 'DT', 'WDT', 'MD', 'VB'], ['and', 'its', 'this', 'that', 'will', 'serve']), (['VBG', 'TO', 'VB'], ['attempting', 'to', 'learn']), (['VBD', 'DT', 'NN', ',', 'VB', 'PRP', 'VB'], ['said', 'the', 'fly', ',', 'let', 'us', 'flee']), (['RB', 'VBZ', 'NN', 'IN', 'DT', 'NN', 'TO', 'VB'], ['now', 'im', 'homeless', 'with', 'no', 'place', 'to', 'stay']), (['TO', 'DT', 'NN', 'NN', 'VBD', 'TO', 'VB'], ['to', 'an', 'art', 'class', 'decided', 'to', 'go']), (['DT', 'JJ', 'NN', 'IN', 'NN', 'TO', 'VB'], ['a', 'new', 'value', 'of', 'pi', 'to', 'assign']), (['NNS', 'VBP', 'VBP', 'VB'], ['limericks', 'i', 'cannot', 'compose']), (['CC', 'JJ', 'NN', 'IN', 'PRP', 'RB', 'VB'], ['but', 'i', 'guess', 'that', 'you', 'probably', 'gnu']), (['PRP$', 'NN', 'VBD', 'IN', 'TO', 'VB'], ['my', 'neighbor', 'came', 'over', 'to', 'say']), (['PRP', 'VBD', 'PRP', 'TO', 'VB'], ['they', 'laid', 'him', 'to', 'rest']), (['WP', 'NN', 'MD', 'RB', 'VB'], ['whom', 'nothing', 'could', 'ever', 'embarrass']), (['NNS', 'RB', 'VBD', 'CC', 'VB'], ['im', 'really', 'determined', 'and', 'keen']), (['JJ', 'VBP', 'WRB', 'NN', 'PRP', 'VB'], ['i', 'wonder', 'why', 'didnt', 'it', 'fall']), (['WP$', 'NNS', 'RB', 'MD', 'VB'], ['whose', 'limericks', 'never', 'would', 'scan'])], 'JJ': [(['EX', 'VBD', 'DT', 'JJ', 'NN', 'RB', 'JJ'], ['there', 'was', 'a', 'young', 'man', 'so', 'benighted']), (['IN', 'IN', 'VBN', 'VBN', 'RB', 'JJ'], ['as', 'if', 'hed', 'been', 'really', 'invited']), (['IN', 'CD', 'IN', 'NN', 'NN', 'VBZ', 'RB', 'JJ'], ['for', 'four', 'for', 'eos', 'eos', 'is', 'too', 'many']), (['WP$', 'NN', 'VBD', 'RBS', 'RB', 'JJ'], ['whose', 'nose', 'was', 'most', 'awfully', 'bent']), (['NN', 'RB', 'VBP', 'NNS', 'IN', 'DT', 'JJ'], ['id', 'rather', 'have', 'ears', 'than', 'a', 'nose']), (['PRP$', 'NNS', 'VBD', 'RB', 'JJ'], ['her', 'thanks', 'were', 'so', 'cold']), (['RB', 'JJ', 'CC', 'JJ'], ['so', 'exclusive', 'and', 'few']), (['IN', 'DT', 'NN', 'VBD', 'JJ', 'CC', 'JJ'], ['that', 'no', 'one', 'was', 'present', 'but', 'smarty']), (['IN', 'DT', 'JJ', 'NN', 'IN', 'NN', 'VBD', 'JJ'], ['by', 'a', 'new', 'way', 'of', 'thinking', 'were', 'smitten']), (['PRP', 'IN', 'JJ', 'VBD', 'RB', 'JJ'], ['it', 'at', 'last', 'grew', 'so', 'small']), (['DT', 'NN', 'VBD', ',', 'RB', 'JJ'], ['the', 'teacher', 'said', ',', 'nnot', 'right']), (['IN', 'DT', 'NNS', 'DT', 'JJ'], ['for', 'the', 'patterns', 'all', 'wrong']), (['NN', 'CC', 'DT', 'NNS', 'RB', 'RB', 'JJ'], ['\\(', 'or', 'the', 'papers', 'too', 'long', '\\)']), (['CC', 'VBD', 'NN', 'JJR', 'JJ'], ['and', 'bought', 'something', 'less', 'flash']), (['VB', 'WRB', 'RB', 'JJ'], ['remember', 'when', 'nearly', 'sixteen']), (['RB', 'VB', 'NN', 'PRP', 'VBP', 'JJ'], ['then', 'i', 'bet', 'you', 'cant', 'guess']), (['CC', 'DT', 'CD', 'VBD', 'JJ'], ['but', 'this', 'one', 'was', 'easy']), (['NN', 'RB', 'VBD', 'JJ'], ['i', 'only', 'felt', 'queasy']), (['CC', 'PRP$', 'NNS', 'VBP', 'VBG', 'CC', 'JJ'], ['but', 'my', 'muscles', 'are', 'aching', 'and', 'torn']), (['DT', 'VBZ', 'RB', 'JJ'], ['this', 'limericks', 'simply', 'sublime']), (['PRP', 'VBZ', 'CC', 'JJ'], ['it', 'expresses', 'but', 'nought']), (['WP', 'VBD', 'RB', 'RB', 'JJ'], ['who', 'was', 'so', 'excessively', 'thin']), (['DT', 'NN', 'NN', 'VBD', 'VBD', 'RB', 'JJ'], ['the', 'replacement', 'i', 'bought', 'was', 'too', 'tall']), (['CC', 'RB', 'DT', 'JJ', 'NN', 'VBZ', 'RB', 'JJ'], ['and', 'now', 'the', 'dumb', 'thing', 'is', 'too', 'small'])], 'VBN': [(['PRP', 'RB', 'VBD', 'WRB', 'PRP', 'VBD', 'VBN'], ['he', 'never', 'knew', 'when', 'he', 'was', 'slighted']), (['IN', 'DT', 'JJ', 'NN', 'VBD', 'VBN'], ['in', 'a', 'funeral', 'procession', 'was', 'spied']), (['PRP', 'VBD', 'VBN', 'PRP', 'MD', 'VB', 'VBN'], ['she', 'was', 'frightened', 'it', 'must', 'be', 'allowed']), (['CC', 'DT', 'NN', 'MD', 'VB', 'VBN'], ['but', 'no', 'horse', 'could', 'be', 'found']), (['CC', 'VBZ', 'PRP', 'VBZ', 'VB', 'VBN'], ['and', 'sideways', 'he', 'couldnt', 'be', 'seen']), (['WRB', 'IN', 'JJS', 'PRP', 'VBD', 'VBN'], ['when', 'at', 'least', 'she', 'was', 'wed']), (['RB', 'DT', 'NN', 'IN', 'DT', 'NN', 'RB', 'VBN'], ['yet', 'the', 'end', 'of', 'the', 'storys', 'not', 'written']), (['TO', 'DT', 'NN', 'IN', 'DT', 'JJ', 'VBG', 'VBN'], ['to', 'the', 'sound', 'of', 'a', 'tooth', 'being', 'filled']), (['JJ', 'RB', 'VBN', 'IN', 'DT', 'NN', 'NN', 'VBD', 'VBN'], ['ive', 'not', 'used', 'since', 'the', 'year', 'i', 'was', 'born']), (['VB', 'RB', ',', 'VBZ', 'VB', 'VBN'], ['come', 'away', ',', 'lets', 'be', 'wed']), (['IN', 'PRP$', 'NNS', 'VBN'], ['because', 'its', 'feet', 'stuck'])], ',': [(['CC', 'VB', 'RB', 'IN', 'NN', ','], ['and', 'eat', 'just', 'as', 'hearty', ',']), (['DT', 'NN', 'IN', 'NN', ',', 'JJ', 'NN', ','], ['a', 'maiden', 'at', 'college', ',', 'miss', 'breeze', ',']), (['VBN', 'RP', 'IN', 'NN', 'DT', 'NN', 'CC', 'JJ', 'NN', ','], ['weighed', 'down', 'by', 'b', 'a', 's', 'and', 'lit', 'ds', ',']), (['VBN', 'IN', 'DT', 'NN', ','], ['collapsed', 'from', 'the', 'strain', ',']), (['DT', 'NN', ',', 'WP', 'VBD', 'IN', 'JJ', 'NN', ','], ['a', 'painter', ',', 'who', 'lived', 'in', 'great', 'britain', ',']), (['PRP', 'VBD', ',', 'IN', 'DT', 'NN', ','], ['he', 'said', ',', 'with', 'a', 'sigh', ',']), (['DT', 'NN', ',', 'RB', 'JJ', ','], ['a', 'canner', ',', 'exceedingly', 'canny', ',']), (['CD', 'NN', 'VBD', 'TO', 'PRP$', 'NN', ','], ['one', 'morning', 'remarked', 'to', 'his', 'granny', ',']), (['NNS', 'TO', 'DT', 'NN', ','], ['heres', 'to', 'the', 'chigger', ',']), (['JJ', 'NNS', 'IN', 'NNS', ','], ['sure', 'itches', 'like', 'blazes', ',']), (['PRP', 'VBD', 'CC', 'VBD', ','], ['he', 'giggled', 'and', 'said', ',']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'IN', 'NN', ','], ['there', 'was', 'a', 'young', 'lady', 'of', 'cork', ',']), (['TO', 'VB', 'RP', 'DT', 'NN', ','], ['to', 'scare', 'off', 'the', 'critter', ',']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'VBN', 'NN', ','], ['there', 'was', 'a', 'young', 'woman', 'named', 'kite', ',']), (['WP$', 'NN', 'VBD', 'RB', 'JJR', 'IN', 'NN', ','], ['whose', 'speed', 'was', 'much', 'faster', 'than', 'light', ',']), (['PRP', 'VBD', 'RP', 'CD', 'NN', ','], ['she', 'set', 'out', 'one', 'day', ',']), (['IN', 'DT', 'JJ', 'NN', ','], ['in', 'a', 'relative', 'way', ',']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'IN', 'NN', ','], ['there', 'was', 'a', 'dear', 'lady', 'of', 'eden', ',']), (['PRP', 'VBD', 'CD', 'TO', 'VB', ','], ['she', 'gave', 'one', 'to', 'adam', ',']), (['WP', 'VBD', ',', 'NN', 'PRP', ',', 'NNS', ','], ['who', 'said', ',', 'thank', 'you', ',', 'madam', ',']), (['VBD', 'DT', 'JJ', ',', 'JJ', 'NN', ','], ['said', 'an', 'envious', ',', 'crudite', 'ermine', ',']), (['WRB', 'DT', 'NN', 'VBZ', 'PRP$', 'NN', ','], ['when', 'a', 'dame', 'wears', 'my', 'coat', ',']), (['DT', 'NN', 'CC', 'DT', 'NN', 'IN', 'DT', 'NN', ','], ['a', 'flea', 'and', 'a', 'fly', 'in', 'a', 'flue', ',']), (['VB', 'PRP', 'VB', ',', 'VBD', 'DT', 'NN', ','], ['let', 'us', 'fly', ',', 'said', 'the', 'flea', ',']), (['VBD', 'DT', 'CD', 'TO', 'DT', 'NN', ','], ['said', 'the', 'two', 'to', 'the', 'tutor', ',']), (['DT', 'JJ', ',', 'IN', 'JJ', 'NN', ','], ['a', 'major', ',', 'with', 'wonderful', 'force', ',']), (['PDT', 'DT', 'NNS', 'VBD', 'NN', ','], ['all', 'the', 'flowers', 'looked', 'round', ',']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'VBN', 'JJ', ','], ['there', 'was', 'an', 'old', 'fellow', 'named', 'green', ',']), (['WP', 'VBD', 'RB', 'RB', 'JJ', ','], ['who', 'grew', 'so', 'abnormally', 'lean', ',']), (['CC', 'JJ', ',', 'CC', 'VBD', ','], ['and', 'flat', ',', 'and', 'compressed', ',']), (['IN', 'PRP$', 'NN', 'VBD', 'PRP$', 'NN', ','], ['that', 'his', 'back', 'touched', 'his', 'chest', ',']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'VBN', 'NN', ','], ['there', 'was', 'a', 'young', 'lady', 'named', 'hannah', ',']), (['IN', 'PRP', 'VBD', 'IN', 'PRP', 'NN', ','], ['as', 'she', 'lay', 'on', 'her', 'side', ',']), (['DT', 'NN', 'VBD', 'RB', 'IN', 'PRP$', 'NN', ','], ['the', 'sultan', 'got', 'sore', 'on', 'his', 'harem', ',']), (['PRP', 'VBD', 'RB', 'JJ', ','], ['she', 'ran', 'almost', 'flew', ',']), (['PRP$', 'NN', 'VBD', 'RB', ','], ['her', 'complexion', 'did', 'too', ',']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'IN', 'DT', 'NN', ','], ['there', 'was', 'an', 'old', 'man', 'in', 'a', 'hearse', ',']), (['VBZ', 'RB', 'JJ', ','], ['is', 'simply', 'immense', ',']), (['VBD', 'DT', 'NN', 'IN', 'DT', 'NN', ','], ['said', 'the', 'man', 'at', 'the', 'door', ',']), (['RB', 'CD', 'IN', 'NN', 'NN', ','], ['not', 'four', 'for', 'eos', 'eos', ',']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'IN', 'NN', ','], ['there', 'was', 'a', 'young', 'lady', 'of', 'kent', ',']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'IN', 'NN', ','], ['there', 'was', 'a', 'young', 'lady', 'of', 'lynn', ',']), (['WP', 'VBD', 'RB', 'RB', 'JJ', ','], ['who', 'was', 'so', 'excessively', 'thin', ',']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'VBN', 'MD', ','], ['there', 'was', 'a', 'young', 'lady', 'named', 'may', ',']), (['PRP$', 'NN', ',', 'PRP', 'VBD', ','], ['its', 'funny', ',', 'she', 'said', ',']), (['DT', 'NN', 'CC', 'PRP$', 'NN', 'NN', ',', 'NN', ','], ['a', 'man', 'and', 'his', 'lady', 'love', ',', 'min', ',']), (['VBN', 'RP', 'WRB', 'DT', 'NN', 'VBD', 'RB', 'JJ', ','], ['skated', 'out', 'where', 'the', 'ice', 'was', 'quite', 'thin', ',']), (['VBD', 'DT', 'NN', ',', 'DT', 'NN', ','], ['had', 'a', 'quarrel', ',', 'no', 'doubt', ',']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'IN', 'NN', ','], ['there', 'was', 'a', 'young', 'lady', 'of', 'munich', ',']), (['WP$', 'NN', 'RB', 'VBD', 'JJ', ','], ['whose', 'appetite', 'simply', 'was', 'unich', ',']), (['NNS', 'NN', 'IN', 'DT', 'NN', ','], ['theres', 'nothing', 'like', 'a', 'food', ',']), (['PRP', 'RB', 'VBD', ','], ['she', 'contentedly', 'cooed', ',']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'IN', 'NN', ','], ['there', 'was', 'a', 'young', 'lady', 'from', 'niger', ',']), (['IN', 'DT', 'NN', 'RB', ','], ['with', 'the', 'lady', 'inside', ',']), (['EX', 'RB', 'VBD', 'DT', 'NN', 'VBN', 'NN', ','], ['there', 'once', 'was', 'a', 'guy', 'named', 'othello', ',']), (['IN', 'VBG', 'PRP$', 'NN', ','], ['after', 'croaking', 'his', 'wife', ',']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'VBN', 'NNS', ','], ['there', 'was', 'a', 'young', 'lady', 'named', 'perkins', ',']), (['IN', 'NN', 'IN', 'NN', ','], ['in', 'spite', 'of', 'advice', ',']), (['PRP', 'VBD', 'RB', 'JJ', 'NN', ','], ['she', 'ate', 'so', 'much', 'spice', ',']), (['IN', 'DT', 'NN', 'NN', 'VBP', 'RB', 'DT', 'NN', ','], ['as', 'a', 'beauty', 'i', 'am', 'not', 'a', 'star', ',']), (['CC', 'PRP$', 'NN', 'NN', 'VBP', 'NN', 'PRP', ','], ['but', 'my', 'face', 'i', 'dont', 'mind', 'it', ',']), (['TO', 'VB', 'DT', 'NN', 'NN', ','], ['to', 'compose', 'a', 'sonata', 'today', ',']), (['IN', 'PRP$', 'NNS', 'IN', 'DT', 'NNS', ','], ['with', 'your', 'toes', 'on', 'the', 'keys', ',']), (['RB', 'VBZ', 'DT', 'JJ', 'NN', 'VBN', 'NN', ','], ['here', 'lies', 'a', 'young', 'salesman', 'named', 'phipps', ',']), (['WP', 'VBD', 'IN', 'CD', 'IN', 'PRP$', 'NNS', ','], ['who', 'married', 'on', 'one', 'of', 'his', 'trips', ',']), (['DT', 'NN', 'VBN', 'NN', ','], ['a', 'widow', 'named', 'block', ',']), (['RB', 'VBD', 'IN', 'DT', 'NN', ','], ['then', 'died', 'of', 'the', 'shock', ',']), (['CC', 'IN', 'IN', 'PRP$', 'NN', ','], ['and', 'as', 'for', 'my', 'hair', ',']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'VBN', 'NN', ','], ['there', 'was', 'a', 'young', 'fellow', 'named', 'weir', ',']), (['WRB', 'PRP', 'VBD', 'PRP', ',', 'PRP', 'VBP', ','], ['when', 'it', 'bored', 'him', ',', 'you', 'know', ',']), (['TO', 'VB', 'TO', 'CC', 'VB', ','], ['to', 'walk', 'to', 'and', 'fro', ',']), (['PRP', 'VBD', ',', 'JJ', 'VBD', ','], ['they', 'quarreled', ',', 'im', 'told', ',']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'VBN', 'NN', ','], ['there', 'was', 'a', 'young', 'person', 'called', 'smarty', ',']), (['PRP', 'RB', 'VBD', ','], ['he', 'gladly', 'proclaimed', ',']), (['PRP$', 'NNS', 'VBP', 'DT', 'JJ', ','], ['their', 'rates', 'are', 'the', 'same', ',']), (['PRP', 'VBZ', ',', 'JJ', 'NN', ','], ['it', 'goes', ',', 'i', 'declare', ',']), (['IN', 'JJ', 'NNS', 'VBD', ','], ['as', 'spacetime', 'thats', 'curved', ',']), (['PRP', 'VBP', 'CC', 'PRP', 'VBP', ','], ['we', 'twist', 'and', 'we', 'turn', ',']), (['IN', 'PRP$', 'NN', 'VBD', 'NN', ','], ['since', 'her', 'apple', 'cheeked', 'groom', ',']), (['IN', 'CD', 'NNS', 'IN', 'DT', 'NN', ','], ['with', 'three', 'wives', 'in', 'the', 'tomb', ',']), (['DT', 'NN', 'CC', 'DT', 'NN', 'IN', 'DT', 'NN', ','], ['a', 'flea', 'and', 'a', 'fly', 'in', 'a', 'flue', ',']), (['NN', 'VBD', 'DT', 'JJ', 'NN', 'NN', ','], ['i', 'bought', 'a', 'new', 'hoover', 'today', ',']), (['VBD', 'PRP', 'IN', 'IN', 'DT', 'JJ', 'NN', ','], ['plugged', 'it', 'in', 'in', 'the', 'usual', 'way', ',']), (['PRP', 'VBD', 'NN', 'IN', ','], ['it', 'sucked', 'everything', 'in', ',']), (['EX', 'RB', 'VBD', 'DT', 'JJ', 'NN', 'IN', 'NN', ','], ['there', 'once', 'was', 'an', 'old', 'man', 'of', 'esser', ',']), (['WP$', 'NN', 'VBD', 'JJR', 'CC', 'JJR', ','], ['whose', 'knowledge', 'grew', 'lesser', 'and', 'lesser', ',']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'IN', 'NN', ','], ['there', 'was', 'a', 'young', 'lady', 'from', 'hyde', ',']), (['IN', 'PRP$', 'NN', 'VBD', ','], ['while', 'her', 'lover', 'lamented', ',']), (['DT', 'NN', 'VBD', ','], ['the', 'apple', 'fermented', ',']), (['DT', 'JJ', 'JJ', 'NN', 'VBN', 'NN', ','], ['an', 'artistic', 'young', 'man', 'called', 'bo', ',']), (['NN', 'VBD', ',', 'CC', 'PRP', 'VBD', ','], ['i', 'enquired', ',', 'but', 'he', 'said', ',']), (['NN', 'DT', 'JJ', 'NN', 'IN', 'NN', ','], ['tis', 'a', 'favorite', 'project', 'of', 'mine', ',']), (['NN', 'MD', 'VB', 'PRP', 'IN', 'NN', ','], ['i', 'would', 'fix', 'it', 'at', 'eos', ',']), (['IN', 'PRP$', 'NN', ',', 'PRP', 'VBP', ','], ['for', 'its', 'simpler', ',', 'you', 'see', ',']), (['EX', 'RB', 'VBD', 'DT', 'NN', 'VBN', 'NN', ','], ['there', 'once', 'was', 'a', 'girl', 'named', 'irene', ',']), (['DT', 'JJ', 'NN', ','], ['a', 'new', 'hydrocarbon', ',']), (['DT', 'NN', 'VBD', 'VBN', 'TO', 'VB', ','], ['a', 'mosquito', 'was', 'heard', 'to', 'complain', ',']), (['NNS', 'VBP', 'CC', 'VB', '.', 'NN', ','], ['thats', 'odd', 'and', 'peculiar', '?', 'funny', ',']), (['PRP', 'VBZ', 'RB', ',', 'PRP', 'VBP', ','], ['it', 'goes', 'quickly', ',', 'you', 'know', ',']), (['CD', 'NN', 'CC', 'DT', 'NN', ','], ['two', 'owls', 'and', 'a', 'hen', ',']), (['IN', 'DT', 'NNS', '.', 'IN', 'UH', ','], ['at', 'the', 'movies', '?', 'if', 'yes', ',']), (['JJ', 'TO', 'RB', 'TO', 'NN', 'NN', ','], ['due', 'to', 'up', 'to', 'date', 'science', ',']), (['TO', 'JJS', 'IN', 'PRP$', 'NNS', ','], ['to', 'most', 'of', 'his', 'clients', ',']), (['VBD', 'DT', 'JJ', ',', 'JJ', 'NN', ','], ['said', 'an', 'envious', ',', 'erudite', 'ermine', ',']), (['WRB', 'DT', 'NN', 'VBZ', 'PRP$', 'NN', ','], ['when', 'a', 'girl', 'wears', 'my', 'coat', ',']), (['IN', 'NNS', 'VBP', 'RB', ',', 'PRP', 'VBD', ','], ['if', 'i', 'wake', 'up', ',', 'he', 'said', ',']), (['IN', 'DT', 'NN', 'IN', 'PRP$', 'NN', ','], ['with', 'a', 'hat', 'on', 'my', 'head', ',']), (['CC', 'VB', 'PRP', 'IN', ','], ['and', 'wave', 'it', 'about', ',']), (['RB', 'VBN', 'VBN', 'IN', 'DT', 'NN', ','], ['theyd', 'been', 'laid', 'on', 'a', 'chair', ',']), (['VBN', 'VBD', 'PRP', 'VBD', 'RB', ','], ['hed', 'forgot', 'they', 'were', 'there', ',']), (['NN', 'MD', 'VB', 'PRP', 'VB', 'VBP', ','], ['i', 'will', 'do', 'it', 'i', 'say', ',']), (['JJ', 'VBN', 'PRP', 'JJ', 'VBN', 'PDT', 'DT', 'NN', ','], ['ive', 'done', 'it', 'ive', 'done', 'mown', 'the', 'lawn', ',']), (['VBD', 'DT', 'NN', 'IN', 'PRP', 'VBZ', 'IN', 'PRP$', 'NN', ','], ['said', 'an', 'ape', 'as', 'he', 'swung', 'by', 'his', 'tail', ',']), (['TO', 'PRP$', 'VBG', 'DT', 'JJ', 'CC', 'NN', ','], ['to', 'his', 'offspring', 'both', 'female', 'and', 'male', ',']), (['IN', 'PRP$', 'NN', ',', 'PRP$', 'NNS', ','], ['from', 'your', 'offspring', ',', 'my', 'dears', ',']), (['IN', 'DT', 'NN', 'IN', 'NNS', ','], ['in', 'a', 'couple', 'of', 'years', ',']), (['CC', 'PRP', 'VBD', 'IN', 'PRP', 'VBD', ','], ['and', 'he', 'beamed', 'as', 'he', 'said', ',']), (['PRP$', 'NN', 'VBD', ','], ['her', 'appearance', 'improved', ',']), (['RBR', 'JJ', 'IN', 'IN', 'RB', ','], ['more', 'bizarre', 'though', 'by', 'far', ',']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'IN', 'NN', ','], ['there', 'was', 'an', 'old', 'man', 'from', 'milan', ',']), (['WRB', 'VBD', 'DT', 'VBD', 'RB', ','], ['when', 'told', 'this', 'was', 'so', ','])], '.': [(['PRP', 'VBP', 'VBG', 'PRP', 'IN', 'NNS', '.'], ['you', 'are', 'killing', 'yourself', 'by', 'degrees', '!']), (['CC', 'DT', 'NN', 'NN', 'MD', 'DT', 'MD', ',', 'MD', 'PRP', '.'], ['but', 'a', 'canner', 'cant', 'can', 'a', 'can', ',', 'can', 'he', '?']), (['CC', 'NNS', 'WRB', 'DT', 'NN', 'VBZ', 'IN', '.'], ['and', 'thats', 'where', 'the', 'rub', 'comes', 'in', '!']), (['WRB', 'VBN', ',', 'WP', 'VBZ', 'JJ', '.'], ['when', 'asked', ',', 'who', 'is', 'dead', '?']), (['WRB', 'NN', 'VBP', 'PRP', ',', 'VBZ', 'VBN', 'DT', 'NN', '.'], ['when', 'i', 'wear', 'it', ',', 'im', 'called', 'a', 'vermine', '!']), (['VBD', 'VBN', ',', 'IN', 'WP', 'MD', 'PRP', 'VB', '.'], ['were', 'imprisoned', ',', 'so', 'what', 'could', 'they', 'do', '?']), (['VBD', 'DT', 'NN', ',', 'VB', 'PRP', 'VB', '.'], ['said', 'the', 'fly', ',', 'let', 'us', 'flee', '!']), (['TO', 'VB', 'CD', 'NNS', 'TO', 'VB', '.'], ['to', 'tutor', 'two', 'tutors', 'to', 'toot', '?']), (['CD', 'NNS', 'RB', 'VBP', 'VBP', 'PRP', 'DT', '.'], ['four', 'tickets', 'ill', 'take', 'have', 'you', 'any', '?']), (['WP', 'DT', 'NN', 'PRP', 'VBP', 'NN', 'IN', '.'], ['what', 'a', 'blessing', 'they', 'didnt', 'fall', 'in', '!']), (['DT', 'NN', 'VBD', 'JJ', ',', 'PRP', 'VBD', 'JJ', '.'], ['that', 'bird', 'wasnt', 'black', ',', 'he', 'was', 'yellow', '!']), (['CC', 'RB', 'JJ', 'IN', 'NN', '.'], ['and', 'quite', 'independent', 'of', 'girth', '!']), (['DT', 'VBG', 'NN', 'IN', 'NN', '.'], ['the', 'superstring', 'theory', 'of', 'witten', '!']), (['VBD', 'VBN', ',', 'IN', 'WP', 'MD', 'PRP', 'VB', '.'], ['were', 'caught', ',', 'so', 'what', 'could', 'they', 'do', '?']), (['WRB', 'VBN', ',', 'WRB', 'RB', 'JJ', '.'], ['when', 'asked', ',', 'why', 'so', 'blue', '?']), (['PRP$', 'NN', 'VBZ', 'DT', 'JJ', '.'], ['your', 'page', 'is', 'all', 'white', '!']), (['VB', 'PRP', 'JJ', 'CC', 'NN', '.'], ['have', 'you', 'needle', 'and', 'thread', '?']), (['CC', 'IN', 'RB', 'VBZ', 'RB', 'VBN', '.'], ['and', 'since', 'then', 'has', 'never', 'benzene', '!']), (['DT', 'NN', 'VBZ', 'VBN', 'PRP$', 'NN', '.'], ['a', 'chemist', 'has', 'poisoned', 'my', 'brain', '!']), (['WP', 'VBD', ',', 'PRP$', 'RB', 'IN', 'NN', 'VBD', '.'], ['who', 'said', ',', 'its', 'just', 'as', 'i', 'feared', '!']), (['VBP', 'VBG', 'DT', 'NN', 'IN', 'PRP$', 'NN', '.'], ['are', 'making', 'a', 'nest', 'in', 'my', 'beard', '!']), (['NN', 'PRP', 'VBP', 'DT', 'JJ', 'NN', '.'], ['dont', 'you', 'know', 'a', 'good', 'tune', '?']), (['CC', 'VBZ', 'NN', 'VB', 'NNS', 'RB', 'JJ', '.'], ['or', 'does', 'gravity', 'miss', 'things', 'so', 'small', '?']), (['CC', 'VBD', ',', 'FW', ',', 'MD', 'PRP', 'VB', '.'], ['and', 'said', ',', 'miss', ',', 'can', 'we', 'dance', '?'])], 'MD': [(['DT', 'NN', 'MD', 'MD'], ['a', 'canner', 'can', 'can']), (['NN', 'IN', 'PRP', 'MD'], ['anything', 'that', 'he', 'can']), (['CC', 'NN', 'RB', 'VB', 'TO', 'VB', 'IN', 'JJ', 'NNS', 'IN', 'DT', 'JJ', 'NN', 'IN', 'NN', 'RB', 'MD'], ['but', 'i', 'always', 'try', 'to', 'get', 'as', 'many', 'syllables', 'into', 'the', 'last', 'line', 'as', 'i', 'possibly', 'can'])], 'JJR': [(['DT', 'NN', 'VBZ', 'DT', 'JJR'], ['the', 'bug', 'thats', 'no', 'bigger']), (['WP', 'VBD', ',', 'DT', 'MD', 'VB', 'VBN', 'JJR'], ['who', 'murmured', ',', 'this', 'might', 'have', 'been', 'worse'])], 'VBZ': [(['CC', 'DT', 'NN', 'IN', 'PRP', 'VBZ'], ['but', 'the', 'welt', 'that', 'he', 'raises']), (['RB', 'VB', 'RB', 'VBD', ',', 'WRB', 'PRP', 'VBZ'], ['ill', 'be', 'awfully', 'said', ',', 'when', 'it', 'goes'])], 'PRP': [(['DT', 'NN', 'WP', 'VBD', 'PRP'], ['a', 'tutor', 'who', 'taught', 'her']), (['IN', 'NN', 'VBP', 'IN', 'PRP'], ['for', 'i', 'am', 'behind', 'it']), (['VBZ', 'RB', 'RBR', 'IN', 'PRP'], ['is', 'supposedly', 'better', 'for', 'you']), (['RB', 'JJ', 'VBD', 'PRP', 'CC', 'VBD', 'PRP'], ['so', 'i', 'hacked', 'it', 'and', 'chopped', 'it']), (['CC', 'RB', 'VBD', 'PRP'], ['and', 'carefully', 'lopped', 'it'])], 'PRP$': [(['RB', 'DT', 'JJ', 'NN', 'VBD', 'PRP$'], ['soon', 'a', 'happy', 'thought', 'hit', 'her'])], 'VBD': [(['PRP', 'VBD', 'RP', 'IN', 'NN', 'CC', 'VBD'], ['she', 'sat', 'up', 'in', 'bed', 'and', 'meowed']), (['CC', 'DT', 'NN', 'PRP', 'VBD'], ['but', 'the', 'copy', 'he', 'wrote']), (['RBR', 'NNS', 'PRP', 'VBD'], ['more', 'stars', 'she', 'espied']), (['CC', 'DT', 'NN', 'VBZ', 'WDT', 'NN', 'PRP', 'VBD'], ['and', 'no', 'one', 'knows', 'which', 'way', 'she', 'went']), (['DT', 'WRB', 'PRP', 'VBD'], ['that', 'when', 'she', 'assayed']), (['DT', 'NN', 'IN', 'NN', 'IN', 'NN', 'VBD'], ['the', 'bottle', 'of', 'perfume', 'that', 'willie', 'sent']), (['VBD', 'DT', 'NNS', 'IN', 'PRP', 'VBD'], ['were', 'the', 'friends', 'that', 'he', 'knew']), (['WP', 'VBP', 'DT', 'JJ', 'NN', 'CC', 'VBD'], ['who', 'ate', 'a', 'green', 'apple', 'and', 'died']), (['IN', 'NN', 'NN', 'VBD', 'IN', 'NN', 'VBD'], ['at', 'eos', 'i', 'sighed', 'as', 'i', 'hoped']), (['IN', 'DT', 'NN', 'IN', 'DT', 'NN', 'NN', 'VBD'], ['for', 'a', 'bike', 'like', 'a', 'harley', 'i', 'groped']), (['RB', 'VBZ', 'VBP', 'IN', 'PRP$', 'NN', 'CC', 'VBD'], ['then', 'i', 'sat', 'on', 'my', 'moped', 'and', 'moped']), (['PRP', 'VBD', 'DT', 'NN', 'PRP', 'VBD'], ['he', 'wasnt', 'the', 'wizard', 'he', 'woz']), (['CC', 'NN', ',', 'PRP', 'VBD'], ['but', 'shaken', ',', 'he', 'shot']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'WP', 'VBD'], ['there', 'was', 'a', 'young', 'dentist', 'who', 'thrilled']), (['PRP', 'MD', 'VB', ',', 'PRP', 'VBD'], ['he', 'would', 'practise', ',', 'they', 'said']), (['IN', 'DT', 'JJ', 'JJ', 'NN', 'NNS', 'VBD'], ['with', 'the', 'old', 'black', 'decker', 'hes', 'skilled']), (['IN', 'DT', 'NNS', 'IN', 'PRP', 'VBD'], ['til', 'the', 'salts', 'that', 'she', 'shook']), (['IN', 'DT', 'NN', 'IN', 'PRP', 'VBD'], ['in', 'the', 'bath', 'that', 'she', 'took']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'VBN', 'VBD'], ['there', 'was', 'a', 'young', 'lady', 'named', 'rose']), (['WRB', 'PRP', 'VBD', 'PRP', 'VBD'], ['when', 'she', 'had', 'it', 'removed']), (['CC', 'VBD', 'PRP', 'RB', 'VBD'], ['or', 'was', 'it', 'just', 'luck']), (['DT', 'WRB', 'PRP', 'VBD'], ['that', 'when', 'he', 'essayed'])], 'IN': [(['VBD', 'DT', 'NN', 'IN'], ['pulled', 'the', 'fisherman', 'in']), (['PRP', 'VBD', 'IN', 'DT', 'NN', 'CC', 'VBD', 'IN'], ['she', 'slipped', 'through', 'the', 'straw', 'and', 'fell', 'in']), (['NN', 'VBZ', 'RB', 'TO', 'VB', 'RP', 'IN'], ['im', 'eos', 'down', 'to', 'put', 'eos', 'across']), (['WP', 'MD', 'VB', 'TO', 'VB', 'IN', 'PRP$', 'NN', 'IN'], ['who', 'would', 'go', 'to', 'church', 'with', 'his', 'hat', 'on']), (['NN', 'MD', 'VB', 'IN', 'PRP', 'VBD', 'VBN', 'VBN', 'IN'], ['i', 'will', 'know', 'that', 'it', 'hasnt', 'been', 'sat', 'on']), (['DT', 'VBD', 'VB', 'PRP', 'IN'], ['that', 'hed', 'knock', 'me', 'around']), (['IN', 'NN', 'VBN', 'IN'], ['as', 'eos', 'walked', 'by']), (['PRP', 'VBD', 'IN', 'DT', 'NN', 'CC', 'VBD', 'IN'], ['he', 'slipped', 'through', 'the', 'straw', 'and', 'fell', 'in'])], 'VBG': [(['DT', 'NN', 'NN', 'VBD', 'VBG'], ['a', 'newspaper', 'man', 'named', 'fling']), (['EX', 'VBD', 'DT', 'JJ', 'NN', 'IN', 'VBG'], ['there', 'was', 'a', 'young', 'man', 'from', 'dealing']), (['WP', 'VBD', 'DT', 'NN', 'IN', 'VBG'], ['who', 'caught', 'the', 'bus', 'for', 'ealing'])], 'CC': [(['VBZ', 'PRP', 'JJR', 'TO', 'VB', ',', 'CC'], ['is', 'it', 'harder', 'to', 'toot', ',', 'or'])], 'VBP': [(['CC', 'PRP', 'VBD', 'NN', 'RB', 'VBP'], ['and', 'she', 'reached', 'home', 'exceedingly', 'plain']), (['CD', 'NN', ',', 'NN', 'VBP'], ['one', 'day', ',', 'i', 'suppose']), (['DT', ',', 'VB', 'PRP', 'RB', 'VB', 'PRP', ',', 'PRP', 'VBP'], ['no', ',', 'igloo', 'them', 'not', 'sew', 'them', ',', 'you', 'know']), (['IN', 'DT', 'JJ', 'NN', 'NN', 'VBP'], ['of', 'the', 'classical', 'music', 'i', 'play']), (['PRP', 'VBD', 'DT', 'JJ', 'NN', 'VBP', ',', 'VBP'], ['he', 'heard', 'a', 'wee', 'spider', 'say', ',', 'hi']), (['PRP$', 'NN', 'VBD', 'NNS', 'IN', 'NN', 'VBP'], ['its', 'crescent', 'shaped', 'rolls', 'that', 'i', 'want']), (['RB', ',', 'VB', 'VBP', 'PRP', 'NN', ',', 'JJ', 'VBP'], ['well', ',', 'ill', 'do', 'it', 'tomorrow', ',', 'i', 'mean']), (['PRP', 'VBD', ',', 'UH', ',', 'NN', 'VBP'], ['he', 'said', ',', 'yes', ',', 'i', 'know'])], 'RP': [(['IN', 'JJ', 'VBP', 'PRP', 'VBD', 'RP'], ['for', 'i', 'hear', 'they', 'fell', 'out']), (['CC', 'RB', 'VB', 'RB', 'RP'], ['and', 'just', 'scarf', 'eos', 'down'])], 'NNS': [(['WP', 'RB', 'RB', 'VBN', 'IN', 'NNS'], ['who', 'just', 'simply', 'doted', 'on', 'gherkins']), (['IN', 'PRP', 'VBD', 'PRP', 'JJ', 'NNS'], ['that', 'she', 'pickled', 'her', 'internal', 'workins']), (['IN', 'DT', 'NN', 'IN', 'PRP$', 'NNS'], ['bang', 'the', 'floor', 'with', 'your', 'knees']), (['WRB', 'PRP', 'VBD', 'EX', 'VBD', 'CD', 'JJ', 'NNS'], ['when', 'he', 'saw', 'there', 'were', 'six', 'little', 'chips']), (['NN', 'RB', 'VBP', 'NNS', 'IN', 'NNS'], ['id', 'rather', 'have', 'fingers', 'than', 'toes']), (['IN', 'NN', 'VBD', 'VBG', 'PRP$', 'NNS'], ['because', 'i', 'was', 'sniffing', 'my', 'toes']), (['VBD', 'VBN', 'IN', 'NN', 'NNS'], ['got', 'crushed', 'between', 'cylinder', 'blocks']), (['VB', 'PRP$', 'NN', 'IN', 'JJ', 'NNS'], ['mislaid', 'his', 'set', 'of', 'false', 'teeth']), (['CC', 'PRP$', 'NNS', 'VBD', 'RB', 'TO', 'PRP$', 'NNS'], ['but', 'her', 'glasses', 'slipped', 'down', 'to', 'her', 'toes'])], 'RB': [(['EX', 'VBP', 'NNS', 'RBR', 'JJ', 'IN', 'RB'], ['there', 'are', 'others', 'more', 'handsome', 'by', 'far']), (['CC', 'DT', 'NN', 'MD', 'VB', 'VBG', 'CD', ',', 'RB'], ['or', 'the', 'rest', 'will', 'be', 'wanting', 'one', ',', 'too'])], 'EX': [(['NN', 'VB', 'PRP$', 'DT', 'EX'], ['im', 'glad', 'its', 'all', 'there'])], 'TO': [(['PRP', 'VBD', 'PRP', 'CC', 'VBD', 'NNS', 'CC', 'TO'], ['he', 'reversed', 'it', 'and', 'walked', 'fro', 'and', 'to'])], 'DT': [(['PRP', 'VBD', 'NN', 'IN', 'DT'], ['he', 'knew', 'nothing', 'at', 'all']), (['NN', 'MD', 'VB', 'EX', 'VBP', 'DT'], ['i', 'could', 'swear', 'there', 'are', 'some'])], 'FW': [(['IN', 'JJ', 'NN', 'NN', 'FW', 'FW', 'FW', 'FW'], ['than', 'eos', 'point', 'eos', 'eos', 'eos', 'eos', 'eos']), (['VBG', 'RB', 'DT', 'NN', 'IN', 'FW', 'FW'], ['leaving', 'only', 'a', 'pile', 'of', 'de', 'brie'])], 'SYM': [(['NNS', 'IN', 'RB', 'IN', 'DT', 'JJ', 'NN', 'SYM'], ['\\(', 'although', 'not', 'in', 'a', 'neighborly', 'way', '\\)'])], 'CD': [(['CD', 'JJ', 'NN', 'IN', 'CD'], ['one', 'saturday', 'morning', 'at', 'three'])]}

    second_line = {'NN': [(['VBN', 'TO', 'VB', 'DT', 'NN', 'VBN', 'NN'], ['wished', 'to', 'wed', 'a', 'woman', 'named', 'phoebe']), (['VBN', 'CD', 'NNS', 'IN', 'PRP$', 'NN'], ['interrupted', 'two', 'girls', 'with', 'their', 'knittin']), (['WP$', 'NN', 'VBD', 'DT', 'NN', 'IN', 'NN'], ['whose', 'pa', 'made', 'a', 'fortune', 'in', 'pork']), (['WP', 'IN', 'NNS', 'VBD', 'RB', 'NN', 'IN', 'NN'], ['who', 'on', 'apples', 'was', 'quite', 'fond', 'of', 'feedin']), (['NNS', 'CD', 'NN', 'NN', 'VBP', 'NN'], ['theres', 'one', 'thing', 'i', 'cannot', 'determine']), (['RB', 'VBN', 'IN', 'DT', 'NN', 'IN', 'DT', 'NN'], ['once', 'fished', 'from', 'the', 'edge', 'of', 'a', 'fissure']), (['MD', 'VB', 'NN', 'IN', 'DT', 'JJ', 'NN'], ['could', 'make', 'copy', 'from', 'any', 'old', 'thing']), (['VBN', 'RP', 'IN', 'NN', 'NN', 'IN', 'DT', 'NN'], ['called', 'out', 'in', 'hyde', 'park', 'for', 'a', 'horse']), (['WP', 'VBD', 'IN', 'DT', 'NN', 'IN', 'NN'], ['who', 'slipped', 'on', 'a', 'peel', 'of', 'banana']), (['CC', 'VBD', 'DT', 'NN', 'IN', 'TO', 'VB', 'NN'], ['and', 'invented', 'a', 'scheme', 'for', 'to', 'scare', 'em']), (['IN', 'VBG', 'VBD', 'VBN', 'IN', 'DT', 'NN'], ['while', 'walking', 'was', 'caught', 'in', 'the', 'rain']), (['WP', 'VBD', 'DT', 'JJ', 'NN', 'DT', 'NN'], ['who', 'read', 'a', 'love', 'story', 'each', 'day']), (['WP', 'VBD', 'DT', 'PRP$', 'NN', 'IN', 'DT', 'NN'], ['who', 'kept', 'all', 'his', 'cash', 'in', 'a', 'bucket']), (['WP', 'VBD', 'IN', 'PRP', 'VBD', 'IN', 'DT', 'NN'], ['who', 'smiled', 'as', 'she', 'rode', 'on', 'a', 'tiger']), (['DT', 'NN', ',', 'JJ', 'NN'], ['a', 'dark', ',', 'disagreeable', 'fellow']), (['NN', 'NN', 'IN', 'DT', 'JJ', 'JJ', 'NN'], ['dont', 'proceed', 'in', 'the', 'old', 'fashioned', 'way']), (['WP', 'VBD', 'DT', 'NN', 'IN', 'NN'], ['who', 'hadnt', 'an', 'atom', 'of', 'fear']), (['WP', 'VBD', 'RP', 'PRP$', 'NNS', 'IN', 'DT', 'NN'], ['who', 'sent', 'out', 'his', 'cards', 'for', 'a', 'party']), (['PRP$', 'JJ', 'NN', 'IN', 'JJ', 'NN'], ['his', 'own', 'law', 'of', 'gravitys', 'force']), (['VBZ', 'TO', 'VB', 'NN'], ['succeeds', 'to', 'describe', 'gravitation']), (['VBD', 'RB', 'JJ', 'NN'], ['was', 'quite', 'understandable', 'nervis']), (['WP', 'VBD', 'PRP', 'RB', 'IN', 'DT', 'NN'], ['who', 'found', 'himself', 'quite', 'at', 'a', 'loss']), (['CC', 'RB', 'JJ', 'NN', 'VBD', 'DT', 'NN'], ['and', 'quite', 'frankly', 'i', 'havent', 'a', 'clue']), (['VBD', 'RB', 'IN', 'PRP', 'VBD', 'IN', 'DT', 'NN'], ['fell', 'apart', 'as', 'he', 'walked', 'in', 'the', 'snow']), (['WP', 'VBD', 'IN', 'JJ', 'NN'], ['who', 'lived', 'on', 'distilled', 'kerosene']), (['IN', 'PRP$', 'RB', 'JJ', 'NN', 'IN', 'DT', 'NN'], ['on', 'your', 'very', 'first', 'date', 'as', 'a', 'teen']), (['VBN', 'IN', 'PRP$', 'NN', 'NN'], ['retired', 'from', 'his', 'business', 'becoz']), (['NNS', 'CD', 'NN', 'NN', 'VBP', 'NN'], ['theres', 'one', 'thing', 'i', 'cannot', 'determine']), (['IN', 'JJ', 'NNS', 'IN', 'PRP$', 'NN'], ['with', 'noxious', 'smells', 'in', 'my', 'nose']), (['NN', 'DT', 'NN', 'VBD', 'NN'], ['youre', 'an', 'utter', 'uncultured', 'disgrace']), (['VBD', 'DT', 'RB', 'JJ', 'NN', 'IN', 'PRP$', 'NN'], ['found', 'a', 'rather', 'large', 'mouse', 'in', 'his', 'stew']), (['DT', 'NNS', 'NN', 'IN', 'NN'], ['a', 'cheesemongers', 'shop', 'in', 'paree']), (['VB', 'DT', 'RB', ',', 'JJ', 'NN'], ['cooed', 'the', 'shapely', ',', 'urbane', 'debutante']), (['TO', 'VB', 'VBG', 'DT', 'NN', 'DT', 'NN', 'NN'], ['to', 'start', 'giving', 'this', 'house', 'a', 'spring', 'clean']), (['VBD', 'DT', 'PRP', 'NN', 'NN', 'IN', 'DT', 'NN'], ['spied', 'a', 'she', 'melon', 'round', 'as', 'a', 'hoop']), (['WP', 'VBD', 'DT', 'JJ', 'NN', 'IN', 'PRP$', 'NN'], ['who', 'had', 'a', 'large', 'wart', 'on', 'her', 'nose']), (['CC', 'VBG', 'IN', 'NN', 'CC', 'NN'], ['and', 'inspiring', 'in', 'meter', 'and', 'rhyme']), (['VBD', 'DT', 'RB', 'JJ', 'NN', 'IN', 'DT', 'NN'], ['met', 'a', 'pretty', 'young', 'lass', 'in', 'a', 'choir']), (['VBN', 'RP', 'NNS', 'IN', 'DT', 'JJ', 'JJ', 'NN'], ['held', 'up', 'banks', 'in', 'a', 'bright', 'yellow', 'suit'])], 'VBN': [(['PRP', 'RB', 'VBD', 'WRB', 'PRP', 'VBD', 'VBN'], ['he', 'never', 'knew', 'when', 'he', 'was', 'slighted']), (['IN', 'DT', 'JJ', 'NN', 'VBD', 'VBN'], ['in', 'a', 'funeral', 'procession', 'was', 'spied']), (['PRP', 'VBD', 'VBN', 'PRP', 'MD', 'VB', 'VBN'], ['she', 'was', 'frightened', 'it', 'must', 'be', 'allowed']), (['TO', 'DT', 'NN', 'IN', 'DT', 'JJ', 'VBG', 'VBN'], ['to', 'the', 'sound', 'of', 'a', 'tooth', 'being', 'filled'])], ',': [(['VBN', 'RP', 'IN', 'NN', 'DT', 'NN', 'CC', 'JJ', 'NN', ','], ['weighed', 'down', 'by', 'b', 'a', 's', 'and', 'lit', 'ds', ',']), (['CD', 'NN', 'VBD', 'TO', 'PRP$', 'NN', ','], ['one', 'morning', 'remarked', 'to', 'his', 'granny', ',']), (['WP$', 'NN', 'VBD', 'RB', 'JJR', 'IN', 'NN', ','], ['whose', 'speed', 'was', 'much', 'faster', 'than', 'light', ',']), (['WP', 'VBD', 'RB', 'RB', 'JJ', ','], ['who', 'grew', 'so', 'abnormally', 'lean', ',']), (['WP', 'VBD', 'RB', 'RB', 'JJ', ','], ['who', 'was', 'so', 'excessively', 'thin', ',']), (['VBN', 'RP', 'WRB', 'DT', 'NN', 'VBD', 'RB', 'JJ', ','], ['skated', 'out', 'where', 'the', 'ice', 'was', 'quite', 'thin', ',']), (['WP$', 'NN', 'RB', 'VBD', 'JJ', ','], ['whose', 'appetite', 'simply', 'was', 'unich', ',']), (['WP', 'VBD', 'IN', 'CD', 'IN', 'PRP$', 'NNS', ','], ['who', 'married', 'on', 'one', 'of', 'his', 'trips', ',']), (['VBD', 'PRP', 'IN', 'IN', 'DT', 'JJ', 'NN', ','], ['plugged', 'it', 'in', 'in', 'the', 'usual', 'way', ',']), (['WP$', 'NN', 'VBD', 'JJR', 'CC', 'JJR', ','], ['whose', 'knowledge', 'grew', 'lesser', 'and', 'lesser', ',']), (['NNS', 'VBP', 'CC', 'VB', '.', 'NN', ','], ['thats', 'odd', 'and', 'peculiar', '?', 'funny', ',']), (['TO', 'PRP$', 'VBG', 'DT', 'JJ', 'CC', 'NN', ','], ['to', 'his', 'offspring', 'both', 'female', 'and', 'male', ','])], '.': [(['VBD', 'VBN', ',', 'IN', 'WP', 'MD', 'PRP', 'VB', '.'], ['were', 'imprisoned', ',', 'so', 'what', 'could', 'they', 'do', '?']), (['CD', 'NNS', 'RB', 'VBP', 'VBP', 'PRP', 'DT', '.'], ['four', 'tickets', 'ill', 'take', 'have', 'you', 'any', '?']), (['VBD', 'VBN', ',', 'IN', 'WP', 'MD', 'PRP', 'VB', '.'], ['were', 'caught', ',', 'so', 'what', 'could', 'they', 'do', '?']), (['DT', 'NN', 'VBZ', 'VBN', 'PRP$', 'NN', '.'], ['a', 'chemist', 'has', 'poisoned', 'my', 'brain', '!']), (['WP', 'VBD', ',', 'PRP$', 'RB', 'IN', 'NN', 'VBD', '.'], ['who', 'said', ',', 'its', 'just', 'as', 'i', 'feared', '!'])], 'VB': [(['VBN', 'TO', 'VB', 'CD', 'JJ', 'NNS', 'TO', 'VB'], ['tried', 'to', 'teach', 'two', 'young', 'tooters', 'to', 'toot']), (['MD', 'RB', 'VB', 'NN', 'TO', 'VB'], ['could', 'always', 'find', 'something', 'to', 'do']), (['VBD', 'RB', 'VBG', 'TO', 'VB'], ['was', 'highly', 'displeasing', 'to', 'millicent']), (['TO', 'VB', 'PRP$', 'CD', 'NNS', 'VBP', 'TO', 'VB'], ['to', 'watch', 'his', 'two', 'rocks', 'fall', 'to', 'earth']), (['TO', 'DT', 'NN', 'NN', 'VBD', 'TO', 'VB'], ['to', 'an', 'art', 'class', 'decided', 'to', 'go']), (['DT', 'JJ', 'NN', 'IN', 'NN', 'TO', 'VB'], ['a', 'new', 'value', 'of', 'pi', 'to', 'assign']), (['WP', 'NN', 'MD', 'RB', 'VB'], ['whom', 'nothing', 'could', 'ever', 'embarrass']), (['JJ', 'VBP', 'WRB', 'NN', 'PRP', 'VB'], ['i', 'wonder', 'why', 'didnt', 'it', 'fall']), (['WP$', 'NNS', 'RB', 'MD', 'VB'], ['whose', 'limericks', 'never', 'would', 'scan'])], 'JJR': [(['WP', 'VBD', ',', 'DT', 'MD', 'VB', 'VBN', 'JJR'], ['who', 'murmured', ',', 'this', 'might', 'have', 'been', 'worse'])], 'JJ': [(['WP$', 'NN', 'VBD', 'RBS', 'RB', 'JJ'], ['whose', 'nose', 'was', 'most', 'awfully', 'bent']), (['NN', 'RB', 'VBP', 'NNS', 'IN', 'DT', 'JJ'], ['id', 'rather', 'have', 'ears', 'than', 'a', 'nose']), (['IN', 'DT', 'JJ', 'NN', 'IN', 'NN', 'VBD', 'JJ'], ['by', 'a', 'new', 'way', 'of', 'thinking', 'were', 'smitten']), (['CC', 'PRP$', 'NNS', 'VBP', 'VBG', 'CC', 'JJ'], ['but', 'my', 'muscles', 'are', 'aching', 'and', 'torn']), (['WP', 'VBD', 'RB', 'RB', 'JJ'], ['who', 'was', 'so', 'excessively', 'thin']), (['DT', 'NN', 'NN', 'VBD', 'VBD', 'RB', 'JJ'], ['the', 'replacement', 'i', 'bought', 'was', 'too', 'tall'])], 'NNS': [(['WP', 'RB', 'RB', 'VBN', 'IN', 'NNS'], ['who', 'just', 'simply', 'doted', 'on', 'gherkins']), (['VBD', 'VBN', 'IN', 'NN', 'NNS'], ['got', 'crushed', 'between', 'cylinder', 'blocks']), (['VB', 'PRP$', 'NN', 'IN', 'JJ', 'NNS'], ['mislaid', 'his', 'set', 'of', 'false', 'teeth'])], 'RB': [(['EX', 'VBP', 'NNS', 'RBR', 'JJ', 'IN', 'RB'], ['there', 'are', 'others', 'more', 'handsome', 'by', 'far'])], 'VBD': [(['WP', 'VBP', 'DT', 'JJ', 'NN', 'CC', 'VBD'], ['who', 'ate', 'a', 'green', 'apple', 'and', 'died']), (['IN', 'DT', 'NN', 'IN', 'DT', 'NN', 'NN', 'VBD'], ['for', 'a', 'bike', 'like', 'a', 'harley', 'i', 'groped'])], 'IN': [(['WP', 'MD', 'VB', 'TO', 'VB', 'IN', 'PRP$', 'NN', 'IN'], ['who', 'would', 'go', 'to', 'church', 'with', 'his', 'hat', 'on'])], 'PRP': [(['VBZ', 'RB', 'RBR', 'IN', 'PRP'], ['is', 'supposedly', 'better', 'for', 'you'])], 'SYM': [(['NNS', 'IN', 'RB', 'IN', 'DT', 'JJ', 'NN', 'SYM'], ['\\(', 'although', 'not', 'in', 'a', 'neighborly', 'way', '\\)'])], 'VBG': [(['WP', 'VBD', 'DT', 'NN', 'IN', 'VBG'], ['who', 'caught', 'the', 'bus', 'for', 'ealing'])], 'VBP': [(['PRP', 'VBD', 'DT', 'JJ', 'NN', 'VBP', ',', 'VBP'], ['he', 'heard', 'a', 'wee', 'spider', 'say', ',', 'hi'])]}

    third_line = {'VB': [(['CC', ',', 'PRP', 'VBD', ',', 'JJ', 'MD', 'VB'], ['but', ',', 'he', 'said', ',', 'i', 'must', 'see']), (['VBD', 'DT', 'NN', ',', 'VB', 'PRP', 'VB'], ['said', 'the', 'fly', ',', 'let', 'us', 'flee']), (['PRP', 'VBD', 'PRP', 'TO', 'VB'], ['they', 'laid', 'him', 'to', 'rest'])], 'NN': [(['PRP', 'MD', 'VB', 'TO', 'DT', 'NN'], ['he', 'would', 'go', 'to', 'a', 'party']), (['PRP', 'VBD', 'IN', 'PRP$', 'NN'], ['he', 'bought', 'for', 'his', 'daughter']), (['DT', 'NN', 'IN', 'DT', 'NN'], ['a', 'fish', 'with', 'a', 'grin']), (['PRP', 'VBD', 'PRP', 'DT', 'NN'], ['he', 'caught', 'him', 'a', 'mouse']), (['IN', 'NN', 'DT', 'NN'], ['of', 'course', 'the', 'expense']), (['PRP', 'VBD', 'PRP', 'NN'], ['she', 'followed', 'her', 'nose']), (['CC', 'PRP$', 'NN', ',', 'VBN', 'NN'], ['but', 'his', 'daughter', ',', 'named', 'nan']), (['PRP', 'VBD', 'RB', 'IN', 'DT', 'NN'], ['they', 'came', 'back', 'from', 'the', 'ride']), (['PRP', 'VBD', 'DT', 'NN'], ['he', 'indulged', 'a', 'desire']), (['VBD', 'PRP', 'IN', 'WP', 'DT', 'NN'], ['switched', 'it', 'on', 'what', 'a', 'din']), (['CC', 'PRP', 'VBD', 'NN'], ['but', 'she', 'started', 'absorbin']), (['DT', 'NN', 'IN', 'PRP$', 'NN'], ['the', 'cause', 'of', 'his', 'sorrow']), (['CC', 'WRB', 'NNS', 'VBP', 'NN'], ['but', 'when', 'i', 'have', 'dough']), (['IN', 'PRP$', 'NN', 'IN', 'NN'], ['in', 'my', 'pocket', 'for', 'cash']), (['IN', 'DT', 'NN', 'IN', 'NN'], ['than', 'a', 'goulash', 'of', 'rat']), (['NN', 'DT', 'NN', 'NN'], ['youre', 'a', 'simpleton', 'loon']), (['VBD', 'DT', 'NN', ',', 'NN', 'NN'], ['said', 'the', 'waiter', ',', 'dont', 'shout']), (['PRP', 'VBD', 'IN', 'DT', 'NN'], ['it', 'said', 'on', 'the', 'door']), (['VBN', 'TO', 'DT', 'NN'], ['collapsed', 'to', 'the', 'ground']), (['NNS', 'VBD', 'RP', 'TO', 'NN'], ['didnt', 'rush', 'off', 'to', 'town'])], ',': [(['VBN', 'IN', 'DT', 'NN', ','], ['collapsed', 'from', 'the', 'strain', ',']), (['PRP', 'VBD', ',', 'IN', 'DT', 'NN', ','], ['he', 'said', ',', 'with', 'a', 'sigh', ',']), (['PRP', 'VBD', 'RP', 'CD', 'NN', ','], ['she', 'set', 'out', 'one', 'day', ',']), (['PRP', 'VBD', 'CD', 'TO', 'VB', ','], ['she', 'gave', 'one', 'to', 'adam', ',']), (['WRB', 'DT', 'NN', 'VBZ', 'PRP$', 'NN', ','], ['when', 'a', 'dame', 'wears', 'my', 'coat', ',']), (['VBD', 'DT', 'CD', 'TO', 'DT', 'NN', ','], ['said', 'the', 'two', 'to', 'the', 'tutor', ',']), (['PDT', 'DT', 'NNS', 'VBD', 'NN', ','], ['all', 'the', 'flowers', 'looked', 'round', ',']), (['CC', 'JJ', ',', 'CC', 'VBD', ','], ['and', 'flat', ',', 'and', 'compressed', ',']), (['IN', 'PRP', 'VBD', 'IN', 'PRP', 'NN', ','], ['as', 'she', 'lay', 'on', 'her', 'side', ',']), (['PRP', 'VBD', 'RB', 'JJ', ','], ['she', 'ran', 'almost', 'flew', ',']), (['VBD', 'DT', 'NN', 'IN', 'DT', 'NN', ','], ['said', 'the', 'man', 'at', 'the', 'door', ',']), (['PRP$', 'NN', ',', 'PRP', 'VBD', ','], ['its', 'funny', ',', 'she', 'said', ',']), (['VBD', 'DT', 'NN', ',', 'DT', 'NN', ','], ['had', 'a', 'quarrel', ',', 'no', 'doubt', ',']), (['NNS', 'NN', 'IN', 'DT', 'NN', ','], ['theres', 'nothing', 'like', 'a', 'food', ',']), (['IN', 'VBG', 'PRP$', 'NN', ','], ['after', 'croaking', 'his', 'wife', ',']), (['IN', 'NN', 'IN', 'NN', ','], ['in', 'spite', 'of', 'advice', ',']), (['CC', 'PRP$', 'NN', 'NN', 'VBP', 'NN', 'PRP', ','], ['but', 'my', 'face', 'i', 'dont', 'mind', 'it', ',']), (['IN', 'PRP$', 'NNS', 'IN', 'DT', 'NNS', ','], ['with', 'your', 'toes', 'on', 'the', 'keys', ',']), (['DT', 'NN', 'VBN', 'NN', ','], ['a', 'widow', 'named', 'block', ',']), (['CC', 'IN', 'IN', 'PRP$', 'NN', ','], ['and', 'as', 'for', 'my', 'hair', ',']), (['WRB', 'PRP', 'VBD', 'PRP', ',', 'PRP', 'VBP', ','], ['when', 'it', 'bored', 'him', ',', 'you', 'know', ',']), (['PRP', 'RB', 'VBD', ','], ['he', 'gladly', 'proclaimed', ',']), (['PRP', 'VBZ', ',', 'JJ', 'NN', ','], ['it', 'goes', ',', 'i', 'declare', ',']), (['IN', 'JJ', 'NNS', 'VBD', ','], ['as', 'spacetime', 'thats', 'curved', ',']), (['PRP', 'VBP', 'CC', 'PRP', 'VBP', ','], ['we', 'twist', 'and', 'we', 'turn', ',']), (['IN', 'PRP$', 'NN', 'VBD', 'NN', ','], ['since', 'her', 'apple', 'cheeked', 'groom', ',']), (['IN', 'PRP$', 'NN', 'VBD', ','], ['while', 'her', 'lover', 'lamented', ',']), (['NN', 'MD', 'VB', 'PRP', 'IN', 'NN', ','], ['i', 'would', 'fix', 'it', 'at', 'eos', ',']), (['CD', 'NN', 'CC', 'DT', 'NN', ','], ['two', 'owls', 'and', 'a', 'hen', ',']), (['IN', 'DT', 'NNS', '.', 'IN', 'UH', ','], ['at', 'the', 'movies', '?', 'if', 'yes', ',']), (['JJ', 'TO', 'RB', 'TO', 'NN', 'NN', ','], ['due', 'to', 'up', 'to', 'date', 'science', ',']), (['WRB', 'DT', 'NN', 'VBZ', 'PRP$', 'NN', ','], ['when', 'a', 'girl', 'wears', 'my', 'coat', ',']), (['IN', 'NNS', 'VBP', 'RB', ',', 'PRP', 'VBD', ','], ['if', 'i', 'wake', 'up', ',', 'he', 'said', ',']), (['RB', 'VBN', 'VBN', 'IN', 'DT', 'NN', ','], ['theyd', 'been', 'laid', 'on', 'a', 'chair', ',']), (['NN', 'MD', 'VB', 'PRP', 'VB', 'VBP', ','], ['i', 'will', 'do', 'it', 'i', 'say', ',']), (['IN', 'PRP$', 'NN', ',', 'PRP$', 'NNS', ','], ['from', 'your', 'offspring', ',', 'my', 'dears', ',']), (['CC', 'PRP', 'VBD', 'IN', 'PRP', 'VBD', ','], ['and', 'he', 'beamed', 'as', 'he', 'said', ',']), (['RBR', 'JJ', 'IN', 'IN', 'RB', ','], ['more', 'bizarre', 'though', 'by', 'far', ',']), (['WRB', 'VBD', 'DT', 'VBD', 'RB', ','], ['when', 'told', 'this', 'was', 'so', ','])], 'MD': [(['DT', 'NN', 'MD', 'MD'], ['a', 'canner', 'can', 'can'])], '.': [(['WRB', 'VBN', ',', 'WP', 'VBZ', 'JJ', '.'], ['when', 'asked', ',', 'who', 'is', 'dead', '?']), (['VBD', 'DT', 'NN', ',', 'VB', 'PRP', 'VB', '.'], ['said', 'the', 'fly', ',', 'let', 'us', 'flee', '!']), (['WRB', 'VBN', ',', 'WRB', 'RB', 'JJ', '.'], ['when', 'asked', ',', 'why', 'so', 'blue', '?']), (['VB', 'PRP', 'JJ', 'CC', 'NN', '.'], ['have', 'you', 'needle', 'and', 'thread', '?']), (['CC', 'VBD', ',', 'FW', ',', 'MD', 'PRP', 'VB', '.'], ['and', 'said', ',', 'miss', ',', 'can', 'we', 'dance', '?'])], 'PRP$': [(['RB', 'DT', 'JJ', 'NN', 'VBD', 'PRP$'], ['soon', 'a', 'happy', 'thought', 'hit', 'her'])], 'VBD': [(['CC', 'DT', 'NN', 'PRP', 'VBD'], ['but', 'the', 'copy', 'he', 'wrote']), (['DT', 'WRB', 'PRP', 'VBD'], ['that', 'when', 'she', 'assayed']), (['CC', 'NN', ',', 'PRP', 'VBD'], ['but', 'shaken', ',', 'he', 'shot']), (['PRP', 'MD', 'VB', ',', 'PRP', 'VBD'], ['he', 'would', 'practise', ',', 'they', 'said']), (['IN', 'DT', 'NNS', 'IN', 'PRP', 'VBD'], ['til', 'the', 'salts', 'that', 'she', 'shook']), (['WRB', 'PRP', 'VBD', 'PRP', 'VBD'], ['when', 'she', 'had', 'it', 'removed']), (['DT', 'WRB', 'PRP', 'VBD'], ['that', 'when', 'he', 'essayed'])], 'JJ': [(['PRP$', 'NNS', 'VBD', 'RB', 'JJ'], ['her', 'thanks', 'were', 'so', 'cold']), (['RB', 'JJ', 'CC', 'JJ'], ['so', 'exclusive', 'and', 'few']), (['PRP', 'IN', 'JJ', 'VBD', 'RB', 'JJ'], ['it', 'at', 'last', 'grew', 'so', 'small']), (['DT', 'NN', 'VBD', ',', 'RB', 'JJ'], ['the', 'teacher', 'said', ',', 'nnot', 'right']), (['IN', 'DT', 'NNS', 'DT', 'JJ'], ['for', 'the', 'patterns', 'all', 'wrong']), (['CC', 'DT', 'CD', 'VBD', 'JJ'], ['but', 'this', 'one', 'was', 'easy']), (['PRP', 'VBZ', 'CC', 'JJ'], ['it', 'expresses', 'but', 'nought'])], 'IN': [(['DT', 'VBD', 'VB', 'PRP', 'IN'], ['that', 'hed', 'knock', 'me', 'around'])], 'DT': [(['NN', 'MD', 'VB', 'EX', 'VBP', 'DT'], ['i', 'could', 'swear', 'there', 'are', 'some'])], 'VBN': [(['IN', 'PRP$', 'NNS', 'VBN'], ['because', 'its', 'feet', 'stuck'])], 'PRP': [(['RB', 'JJ', 'VBD', 'PRP', 'CC', 'VBD', 'PRP'], ['so', 'i', 'hacked', 'it', 'and', 'chopped', 'it'])]}

    last_two_lines = {'NN': [(['WP', 'DT', 'JJ', 'NN', 'VB', 'IN', 'JJ', 'VB', 'JJ', 'NN', 'NN'], ['what', 'the', 'clerical', 'fee', 'be', 'before', 'phoebe', 'be', 'phoebe', 'bee', 'bee']), (['DT', 'NN', 'NN', 'RB', 'VB', 'RB', 'VBN', 'PRP', ',', 'RB', 'WRB', 'NN', 'NN'], ['that', 'park', 'bench', 'well', 'i', 'just', 'painted', 'it', ',', 'right', 'where', 'youre', 'sittin']), (['PRP', 'VBD', 'CC', 'VBD', ',', 'JJ', 'VBP', 'VBP', 'NN', 'RB', 'VBD', 'IN', 'DT', 'NN'], ['he', 'giggled', 'and', 'said', ',', 'i', 'dont', 'know', 'i', 'just', 'came', 'for', 'the', 'ride']), (['DT', 'NN', 'WP', 'VBD', 'PRP', 'TO', 'VB', 'JJ', 'NNS', 'IN', 'PRP', 'NN'], ['a', 'tutor', 'who', 'taught', 'her', 'to', 'balance', 'green', 'peas', 'on', 'her', 'fork']), (['IN', 'DT', 'JJ', 'NN', ',', 'CC', 'VBN', 'IN', 'DT', 'JJ', 'NN'], ['in', 'a', 'relative', 'way', ',', 'and', 'returned', 'on', 'the', 'previous', 'night']), (['WP', 'VBD', ',', 'NN', 'PRP', ',', 'NNS', ',', 'CC', 'RB', 'DT', 'VBN', 'IN', 'NN'], ['who', 'said', ',', 'thank', 'you', ',', 'madam', ',', 'and', 'then', 'both', 'skedaddled', 'from', 'eden']), (['VBD', 'DT', 'NN', 'IN', 'RB', 'VBZ', 'VBG', 'DT', 'NN', 'IN', 'NN'], ['pulled', 'the', 'fisherman', 'in', 'now', 'theyre', 'fishing', 'the', 'fissure', 'for', 'fisher']), (['IN', 'DT', 'CD', 'NN', 'NN', 'VBD', 'RB', 'JJ', 'PRP', 'VBZ', 'RB', 'IN', 'VBG', 'NN'], ['of', 'a', 'five', 'dollar', 'note', 'was', 'so', 'good', 'he', 'is', 'now', 'in', 'sing', 'sing']), (['VB', 'PRP', 'VB', ',', 'VBD', 'DT', 'NN', ',', 'CC', 'PRP', 'VBD', 'IN', 'DT', 'NN', 'IN', 'DT', 'NN'], ['let', 'us', 'fly', ',', 'said', 'the', 'flea', ',', 'and', 'they', 'flew', 'through', 'a', 'flaw', 'in', 'the', 'flue']), (['CC', 'DT', 'NN', 'MD', 'VB', 'VBN', 'RB', 'PRP', 'RB', 'VB', ',', 'IN', 'NN'], ['but', 'no', 'horse', 'could', 'be', 'found', 'so', 'he', 'just', 'rhododendron', ',', 'of', 'course']), (['RBR', 'NNS', 'PRP', 'VBD', 'IN', 'EX', 'VBP', 'IN', 'DT', 'NN', 'VBD', 'NN'], ['more', 'stars', 'she', 'espied', 'than', 'there', 'are', 'in', 'the', 'star', 'spangled', 'banner']), (['WDT', 'PRP', 'VBD', 'IN', 'DT', 'NN', 'IN', 'DT', 'NN', 'VBZ', 'VBN', 'NN', 'NN', 'NN'], ['which', 'he', 'loosed', 'in', 'the', 'house', '\\(', 'the', 'confusion', 'is', 'called', 'harem', 'scarem', '\\)']), (['VBZ', 'RB', 'JJ', ',', 'CC', 'PRP', 'VBZ', 'VBN', 'IN', 'IN', 'PRP$', 'NN'], ['is', 'simply', 'immense', ',', 'but', 'it', 'doesnt', 'come', 'out', 'of', 'my', 'purse']), (['WRB', 'IN', 'JJS', 'PRP', 'VBD', 'VBN', 'JJ', 'NN', 'VBP', 'NN', 'VBD', 'DT', 'NN'], ['when', 'at', 'least', 'she', 'was', 'wed', 'i', 'didnt', 'think', 'life', 'was', 'this', 'way']), (['PRP', 'RB', 'VBD', ',', 'IN', 'PRP', 'VBD', 'RP', 'CD', 'NNS', 'IN', 'PRP$', 'NN'], ['she', 'contentedly', 'cooed', ',', 'as', 'she', 'let', 'out', 'three', 'tucks', 'in', 'her', 'tunic']), (['VB', 'RP', 'IN', 'DT', 'NN', 'CC', 'RB', 'RB', 'IN', 'DT', 'NN', ',', 'NN'], ['ran', 'away', 'with', 'a', 'man', 'and', 'as', 'far', 'as', 'the', 'bucket', ',', 'nantucket']), (['IN', 'DT', 'NN', 'RB', ',', 'CC', 'DT', 'NN', 'IN', 'DT', 'NN', 'IN', 'DT', 'NN'], ['with', 'the', 'lady', 'inside', ',', 'and', 'the', 'smile', 'on', 'the', 'face', 'of', 'the', 'tiger']), (['IN', 'NN', 'VBP', 'IN', 'PRP', 'PRP$', 'DT', 'NNS', 'IN', 'NN', 'IN', 'JJ', 'NN'], ['for', 'i', 'am', 'behind', 'it', 'its', 'the', 'people', 'in', 'front', 'that', 'i', 'jar']), (['TO', 'VB', 'DT', 'JJ', 'NN', 'RB', 'JJS', 'DT', 'JJ', 'NN', 'MD', 'VB', 'RB', '.', 'NN'], ['to', 'touch', 'a', 'live', 'wire', '\\(', 'most', 'any', 'last', 'line', 'will', 'do', 'here', '!', '\\)']), (['PRP', 'VBD', ',', 'JJ', 'VBD', ',', 'IN', 'DT', 'JJ', 'JJ', 'NN', 'VBD', 'NN'], ['they', 'quarreled', ',', 'im', 'told', ',', 'through', 'that', 'silly', 'scent', 'willie', 'sent', 'millicent']), (['IN', 'DT', 'JJ', 'NN', 'IN', 'DT', 'NN', 'IN', 'NN', 'TO', 'NN'], ['as', 'the', 'inverted', 'square', 'of', 'the', 'distance', 'from', 'object', 'to', 'source']), (['CC', 'PRP$', 'DT', 'WDT', 'MD', 'VB', 'IN', 'DT', 'NNS', 'JJ', 'NN'], ['and', 'its', 'this', 'that', 'will', 'serve', 'as', 'the', 'planets', 'unique', 'motivation']), (['IN', 'CD', 'NNS', 'IN', 'DT', 'NN', ',', 'NN', 'VBG', 'PRP', 'IN', 'DT', 'NN'], ['with', 'three', 'wives', 'in', 'the', 'tomb', ',', 'kept', 'insuring', 'her', 'during', 'the', 'service']), (['VB', 'PRP', 'VB', ',', 'VBD', 'DT', 'NN', 'RB', 'PRP', 'VBP', 'IN', 'DT', 'NN', 'IN', 'DT', 'NN'], ['let', 'us', 'fly', ',', 'said', 'the', 'flea', 'so', 'they', 'flew', 'through', 'a', 'flaw', 'in', 'the', 'flue']), (['PRP', 'VBD', 'NN', 'IN', 'DT', 'CC', 'RB', 'VBZ', 'DT', 'NN', 'NN'], ['he', 'knew', 'nothing', 'at', 'all', 'and', 'now', 'hes', 'a', 'college', 'professor']), (['DT', 'NN', 'VBD', ',', 'CC', 'VBD', 'NN', 'IN', 'PRP$', 'NN'], ['the', 'apple', 'fermented', ',', 'and', 'made', 'cider', 'inside', 'her', 'inside']), (['PRP$', 'NN', 'VBZ', 'DT', 'JJ', '.', 'NN', 'VBD', ',', 'PRP', 'VBZ', 'DT', 'JJ', 'NN', 'IN', 'NN'], ['your', 'page', 'is', 'all', 'white', '!', 'bo', 'said', ',', 'it', 'is', 'a', 'polar', 'bear', 'in', 'snow']), (['NN', 'CC', 'DT', 'NNS', 'RB', 'RB', 'JJ', 'CC', 'VB', 'VBN', 'TO', 'DT', 'NN', 'IN', 'NN'], ['\\(', 'or', 'the', 'papers', 'too', 'long', '\\)', 'and', 'im', 'stuck', 'to', 'the', 'toilet', 'with', 'glue']), (['VBD', 'NN', 'NN'], ['was', 'paradichloro', 'triphenyldichloroethane']), (['PRP', 'VBZ', 'RB', ',', 'PRP', 'VBP', ',', 'CC', 'VBZ', 'IN', 'IN', 'PRP$', 'NNS', 'IN', 'NN'], ['it', 'goes', 'quickly', ',', 'you', 'know', ',', 'and', 'seeps', 'out', 'of', 'my', 'pockets', 'like', 'honey']), (['RB', 'VB', 'NN', 'PRP', 'VBP', 'JJ', 'WP', 'VBD', 'VBN', 'IN', 'DT', 'NN', 'NN'], ['then', 'i', 'bet', 'you', 'cant', 'guess', 'what', 'was', 'shown', 'on', 'the', 'cinema', 'screen']), (['NNS', 'DT', 'NN', 'IN', 'NN', 'WRB', 'NN', 'VBP', 'PRP', ',', 'VB', 'VBN', 'RB', 'NN'], ['shes', 'a', 'person', 'of', 'note', 'when', 'i', 'wear', 'it', ',', 'im', 'called', 'only', 'vermin']), (['NN', 'PRP', 'VBP', 'DT', 'JJ', 'NN', '.', 'RB', 'PRP', 'VBD', 'PRP', 'NN', 'IN', 'DT', 'NN'], ['dont', 'you', 'know', 'a', 'good', 'tune', '?', 'then', 'he', 'walloped', 'me', 'square', 'in', 'the', 'face']), (['NN', 'NN', 'IN', 'DT', 'NN', 'RB', 'PRP', 'VBD', 'RB', 'CC', 'VB', 'IN', 'DT', 'NN'], ['dont', 'spit', 'on', 'the', 'floor', 'so', 'he', 'jumped', 'up', 'and', 'spat', 'on', 'the', 'ceiling']), (['PRP', 'RB', 'RB', 'IN', 'DT', 'NN', 'IN', 'PRP', 'VBD', 'TO', 'VB', ',', 'VB', 'DT', 'NN'], ['it', 'right', 'there', 'on', 'the', 'spot', 'as', 'it', 'tried', 'to', 'explain', ',', 'im', 'a', 'spi']), (['IN', 'PRP$', 'NNS', 'CC', 'PRP$', 'NN', 'IN', 'PRP$', 'NN', 'CC', 'NN', 'IN', 'DT', 'NN'], ['in', 'his', 'boots', 'and', 'his', 'vest', 'with', 'his', 'spanner', 'and', 'jack', 'in', 'the', 'box']), (['CC', 'RB', 'VB', 'RB', 'RP', 'NN', 'VBD', 'WRB', 'NN', 'VBP', 'IN', 'NN'], ['and', 'just', 'scarf', 'eos', 'down', 'i', 'relaxed', 'when', 'i', 'eos', 'across', 'aunt']), (['VBN', 'VBD', 'PRP', 'VBD', 'RB', ',', 'RB', 'RB', ',', 'CC', 'VBD', 'VBN', 'NN'], ['hed', 'forgot', 'they', 'were', 'there', ',', 'sat', 'down', ',', 'and', 'was', 'bitten', 'beneath']), (['IN', 'DT', 'NN', 'IN', 'PRP', 'VBD', 'VBN', 'RP', 'TO', 'VB', 'NN', 'IN', 'NN'], ['in', 'the', 'bath', 'that', 'she', 'took', 'turned', 'out', 'to', 'be', 'plaster', 'of', 'paris']), (['IN', 'DT', 'NN', 'IN', 'NNS', ',', 'MD', 'VB', 'DT', 'NN', 'IN', 'NN'], ['in', 'a', 'couple', 'of', 'years', ',', 'may', 'evolve', 'a', 'professor', 'at', 'yale']), (['VB', 'RB', ',', 'VBZ', 'VB', 'VBN', 'CC', 'PRP', 'VBD', 'CC', 'PRP', 'VBD', ',', 'NN'], ['come', 'away', ',', 'lets', 'be', 'wed', 'but', 'she', 'sighed', 'and', 'she', 'said', ',', 'canteloupe']), (['IN', 'JJ', 'NN', 'CC', 'TO', 'VB', 'PRP', 'JJ', 'NNS', 'IN', 'NN'], ['with', 'intelligent', 'thought', 'and', 'to', 'write', 'it', 'used', 'acres', 'of', 'time']), (['CC', 'PRP', 'VBD', 'PRP', ',', 'DT', 'NN', 'IN', 'JJ', 'VBP', 'IN', 'NN', 'NNS', 'VBP', ',', 'NN'], ['but', 'she', 'told', 'him', ',', 'no', 'chance', 'for', 'i', 'fear', 'that', 'im', 'handels', 'miss', ',', 'sire']), (['PRP', 'MD', 'VB', 'DT', 'NN', 'CC', 'NN', ',', 'NN', 'RB', ',', 'CC', 'RB', 'JJ', 'NN'], ['he', 'would', 'wave', 'a', 'cigar', 'and', 'shout', ',', 'freeze', 'there', ',', 'or', 'else', 'ill', 'cheroot'])], 'JJ': [(['CC', 'VB', 'RB', 'IN', 'NN', ',', 'IN', 'IN', 'VBN', 'VBN', 'RB', 'JJ'], ['and', 'eat', 'just', 'as', 'hearty', ',', 'as', 'if', 'hed', 'been', 'really', 'invited']), (['RB', 'CD', 'IN', 'NN', 'NN', ',', 'IN', 'CD', 'IN', 'NN', 'NN', 'VBZ', 'RB', 'JJ'], ['not', 'four', 'for', 'eos', 'eos', ',', 'for', 'four', 'for', 'eos', 'eos', 'is', 'too', 'many']), (['VBD', 'DT', 'NNS', 'IN', 'PRP', 'VBD', 'IN', 'DT', 'NN', 'VBD', 'JJ', 'CC', 'JJ'], ['were', 'the', 'friends', 'that', 'he', 'knew', 'that', 'no', 'one', 'was', 'present', 'but', 'smarty']), (['CC', 'RB', 'VBD', 'PRP', 'CC', 'RB', 'DT', 'JJ', 'NN', 'VBZ', 'RB', 'JJ'], ['and', 'carefully', 'lopped', 'it', 'and', 'now', 'the', 'dumb', 'thing', 'is', 'too', 'small'])], '.': [(['VBD', 'PRP$', 'NN', ',', 'PRP$', 'NN', 'PRP', 'VBP', 'VBG', 'PRP', 'IN', 'NNS', '.'], ['said', 'her', 'doctor', ',', 'its', 'plain', 'you', 'are', 'killing', 'yourself', 'by', 'degrees', '!']), (['NN', 'IN', 'PRP', 'MD', 'CC', 'DT', 'NN', 'NN', 'MD', 'DT', 'MD', ',', 'MD', 'PRP', '.'], ['anything', 'that', 'he', 'can', 'but', 'a', 'canner', 'cant', 'can', 'a', 'can', ',', 'can', 'he', '?']), (['NNS', 'DT', 'NN', 'IN', 'NN', 'WRB', 'NN', 'VBP', 'PRP', ',', 'VBZ', 'VBN', 'DT', 'NN', '.'], ['shes', 'a', 'person', 'of', 'note', 'when', 'i', 'wear', 'it', ',', 'im', 'called', 'a', 'vermine', '!']), (['VBZ', 'PRP', 'JJR', 'TO', 'VB', ',', 'CC', 'TO', 'VB', 'CD', 'NNS', 'TO', 'VB', '.'], ['is', 'it', 'harder', 'to', 'toot', ',', 'or', 'to', 'tutor', 'two', 'tutors', 'to', 'toot', '?']), (['IN', 'JJ', 'VBP', 'PRP', 'VBD', 'RP', 'WP', 'DT', 'NN', 'PRP', 'VBP', 'NN', 'IN', '.'], ['for', 'i', 'hear', 'they', 'fell', 'out', 'what', 'a', 'blessing', 'they', 'didnt', 'fall', 'in', '!']), (['RB', 'PRP', 'VBD', 'PRP$', 'JJ', 'NN', 'DT', 'NN', 'VBD', 'JJ', ',', 'PRP', 'VBD', 'JJ', '.'], ['then', 'he', 'took', 'his', 'own', 'life', 'that', 'bird', 'wasnt', 'black', ',', 'he', 'was', 'yellow', '!']), (['PRP$', 'NNS', 'VBP', 'DT', 'JJ', ',', 'CC', 'RB', 'JJ', 'IN', 'NN', '.'], ['their', 'rates', 'are', 'the', 'same', ',', 'and', 'quite', 'independent', 'of', 'girth', '!']), (['VBG', 'TO', 'VB', 'DT', 'VBG', 'NN', 'IN', 'NN', '.'], ['attempting', 'to', 'learn', 'the', 'superstring', 'theory', 'of', 'witten', '!']), (['DT', 'JJ', 'NN', ',', 'CC', 'IN', 'RB', 'VBZ', 'RB', 'VBN', '.'], ['a', 'new', 'hydrocarbon', ',', 'and', 'since', 'then', 'has', 'never', 'benzene', '!']), (['CD', 'NNS', 'CC', 'DT', 'NN', 'VBP', 'VBG', 'DT', 'NN', 'IN', 'PRP$', 'NN', '.'], ['four', 'larks', 'and', 'a', 'wren', 'are', 'making', 'a', 'nest', 'in', 'my', 'beard', '!']), (['CC', 'VBD', 'PRP', 'RB', 'VBD', 'CC', 'VBZ', 'NN', 'VB', 'NNS', 'RB', 'JJ', '.'], ['or', 'was', 'it', 'just', 'luck', 'or', 'does', 'gravity', 'miss', 'things', 'so', 'small', '?'])], 'VBD': [(['TO', 'VB', 'RP', 'DT', 'NN', ',', 'PRP', 'VBD', 'RP', 'IN', 'NN', 'CC', 'VBD'], ['to', 'scare', 'off', 'the', 'critter', ',', 'she', 'sat', 'up', 'in', 'bed', 'and', 'meowed']), (['CD', 'NN', ',', 'NN', 'VBP', 'CC', 'DT', 'NN', 'VBZ', 'WDT', 'NN', 'PRP', 'VBD'], ['one', 'day', ',', 'i', 'suppose', 'and', 'no', 'one', 'knows', 'which', 'way', 'she', 'went']), (['CC', 'VBD', 'NN', 'JJR', 'JJ', 'RB', 'VBZ', 'VBP', 'IN', 'PRP$', 'NN', 'CC', 'VBD'], ['and', 'bought', 'something', 'less', 'flash', 'then', 'i', 'sat', 'on', 'my', 'moped', 'and', 'moped']), (['TO', 'JJS', 'IN', 'PRP$', 'NNS', ',', 'PRP', 'VBD', 'DT', 'NN', 'PRP', 'VBD'], ['to', 'most', 'of', 'his', 'clients', ',', 'he', 'wasnt', 'the', 'wizard', 'he', 'woz']), (['DT', 'NN', 'IN', 'PRP$', 'NN', 'IN', 'DT', 'JJ', 'JJ', 'NN', 'NNS', 'VBD'], ['every', 'night', 'in', 'his', 'shed', 'with', 'the', 'old', 'black', 'decker', 'hes', 'skilled'])], 'VBN': [(['IN', 'PRP$', 'NN', 'VBD', 'PRP$', 'NN', ',', 'CC', 'VBZ', 'PRP', 'VBZ', 'VB', 'VBN'], ['that', 'his', 'back', 'touched', 'his', 'chest', ',', 'and', 'sideways', 'he', 'couldnt', 'be', 'seen']), (['IN', 'PRP$', 'NNS', 'CC', 'PRP$', 'NN', 'JJ', 'RB', 'VBN', 'IN', 'DT', 'NN', 'NN', 'VBD', 'VBN'], ['in', 'my', 'legs', 'and', 'my', 'bum', 'ive', 'not', 'used', 'since', 'the', 'year', 'i', 'was', 'born'])], 'VBP': [(['PRP$', 'NN', 'VBD', 'RB', ',', 'CC', 'PRP', 'VBD', 'NN', 'RB', 'VBP'], ['her', 'complexion', 'did', 'too', ',', 'and', 'she', 'reached', 'home', 'exceedingly', 'plain']), (['NN', 'VBD', ',', 'CC', 'PRP', 'VBD', ',', 'DT', ',', 'VB', 'PRP', 'RB', 'VB', 'PRP', ',', 'PRP', 'VBP'], ['i', 'enquired', ',', 'but', 'he', 'said', ',', 'no', ',', 'igloo', 'them', 'not', 'sew', 'them', ',', 'you', 'know']), (['IN', 'JJ', 'VBP', 'VBP', 'DT', 'NN', 'IN', 'DT', 'JJ', 'NN', 'NN', 'VBP'], ['if', 'i', 'didnt', 'curb', 'the', 'sound', 'of', 'the', 'classical', 'music', 'i', 'play']), (['UH', ',', 'JJ', 'VBP', 'PRP', 'NN', 'RB', ',', 'VB', 'VBP', 'PRP', 'NN', ',', 'JJ', 'VBP'], ['yes', ',', 'ill', 'do', 'it', 'today', 'well', ',', 'ill', 'do', 'it', 'tomorrow', ',', 'i', 'mean'])], 'IN': [(['TO', 'VB', 'NN', 'PRP', 'VBD', 'IN', 'DT', 'NN', 'CC', 'VBD', 'IN'], ['to', 'drink', 'lemonade', 'she', 'slipped', 'through', 'the', 'straw', 'and', 'fell', 'in']), (['VBD', ',', 'JJ', 'VBP', 'DT', 'NN', 'NN', 'VBZ', 'RB', 'TO', 'VB', 'RP', 'IN'], ['said', ',', 'i', 'havent', 'a', 'clue', 'im', 'eos', 'down', 'to', 'put', 'eos', 'across']), (['IN', 'DT', 'NN', 'IN', 'PRP$', 'NN', ',', 'NN', 'MD', 'VB', 'IN', 'PRP', 'VBD', 'VBN', 'VBN', 'IN'], ['with', 'a', 'hat', 'on', 'my', 'head', ',', 'i', 'will', 'know', 'that', 'it', 'hasnt', 'been', 'sat', 'on']), (['TO', 'VB', 'NN', 'PRP', 'VBD', 'IN', 'DT', 'NN', 'CC', 'VBD', 'IN'], ['to', 'drink', 'lemonade', 'he', 'slipped', 'through', 'the', 'straw', 'and', 'fell', 'in'])], 'NNS': [(['PRP', 'VBD', 'RB', 'JJ', 'NN', ',', 'IN', 'PRP', 'VBD', 'PRP', 'JJ', 'NNS'], ['she', 'ate', 'so', 'much', 'spice', ',', 'that', 'she', 'pickled', 'her', 'internal', 'workins']), (['RB', 'VBD', 'IN', 'DT', 'NN', ',', 'WRB', 'PRP', 'VBD', 'EX', 'VBD', 'CD', 'JJ', 'NNS'], ['then', 'died', 'of', 'the', 'shock', ',', 'when', 'he', 'saw', 'there', 'were', 'six', 'little', 'chips']), (['NN', 'RB', 'VBD', 'JJ', 'IN', 'NN', 'VBD', 'VBG', 'PRP$', 'NNS'], ['i', 'only', 'felt', 'queasy', 'because', 'i', 'was', 'sniffing', 'my', 'toes']), (['PRP$', 'NN', 'VBD', ',', 'CC', 'PRP$', 'NNS', 'VBD', 'RB', 'TO', 'PRP$', 'NNS'], ['her', 'appearance', 'improved', ',', 'but', 'her', 'glasses', 'slipped', 'down', 'to', 'her', 'toes'])], 'VB': [(['IN', 'DT', 'NN', 'IN', 'PRP$', 'NNS', 'UH', 'WRB', 'JJ', '.', 'DT', 'NNS', 'MD', 'VB'], ['bang', 'the', 'floor', 'with', 'your', 'knees', 'oh', 'how', 'modern', '!', 'the', 'critics', 'will', 'say']), (['PRP', 'VBD', 'NN', 'IN', ',', 'RB', 'VBZ', 'NN', 'IN', 'DT', 'NN', 'TO', 'VB'], ['it', 'sucked', 'everything', 'in', ',', 'now', 'im', 'homeless', 'with', 'no', 'place', 'to', 'stay']), (['CC', 'JJ', 'NN', 'CC', 'JJ', 'NN', 'IN', 'PRP', 'RB', 'VB'], ['or', 'hungarian', 'cat', 'but', 'i', 'guess', 'that', 'you', 'probably', 'gnu'])], 'VBZ': [(['NN', 'VB', 'PRP$', 'DT', 'EX', 'RB', 'VB', 'RB', 'VBD', ',', 'WRB', 'PRP', 'VBZ'], ['im', 'glad', 'its', 'all', 'there', 'ill', 'be', 'awfully', 'said', ',', 'when', 'it', 'goes'])], 'TO': [(['TO', 'VB', 'TO', 'CC', 'VB', ',', 'PRP', 'VBD', 'PRP', 'CC', 'VBD', 'NNS', 'CC', 'TO'], ['to', 'walk', 'to', 'and', 'fro', ',', 'he', 'reversed', 'it', 'and', 'walked', 'fro', 'and', 'to'])], 'FW': [(['IN', 'PRP$', 'NN', ',', 'PRP', 'VBP', ',', 'IN', 'JJ', 'NN', 'NN', 'FW', 'FW', 'FW', 'FW'], ['for', 'its', 'simpler', ',', 'you', 'see', ',', 'than', 'eos', 'point', 'eos', 'eos', 'eos', 'eos', 'eos']), (['IN', 'DT', 'JJ', 'NN', 'VBG', 'RB', 'DT', 'NN', 'IN', 'FW', 'FW'], ['with', 'a', 'thunderous', 'sound', 'leaving', 'only', 'a', 'pile', 'of', 'de', 'brie'])], 'RB': [(['CC', 'VB', 'PRP', 'IN', ',', 'CC', 'DT', 'NN', 'MD', 'VB', 'VBG', 'CD', ',', 'RB'], ['and', 'wave', 'it', 'about', ',', 'or', 'the', 'rest', 'will', 'be', 'wanting', 'one', ',', 'too'])], 'MD': [(['PRP', 'VBD', ',', 'UH', ',', 'NN', 'VBP', 'CC', 'NN', 'RB', 'VB', 'TO', 'VB', 'IN', 'JJ', 'NNS', 'IN', 'DT', 'JJ', 'NN', 'IN', 'NN', 'RB', 'MD'], ['he', 'said', ',', 'yes', ',', 'i', 'know', 'but', 'i', 'always', 'try', 'to', 'get', 'as', 'many', 'syllables', 'into', 'the', 'last', 'line', 'as', 'i', 'possibly', 'can'])]}

    return dataset, second_line, third_line, last_two_lines
