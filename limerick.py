from Generate import *
from functions import *
from Traversal_Glove import *
import random
from gensim.parsing.preprocessing import remove_stopwords
from nltk.corpus import wordnet as wn
import os
from six.moves import cPickle
import requests
from functions import *


class limerick:
    def __init__(self):
        self.gen=Generate(wv_file='./storyline_for_reference/glove.6B.300d.word2vec.txt')
        self.mp = Meta_Poetry_Glove(wv_file='./storyline_for_reference/glove.6B.300d.word2vec.txt')
        #get set of templates
        """if type(template_dataset)==type(None):
            dataset, second_line, third_line, last_two=get_templates()#function in functions.py
        self.dataset=dataset
        self.second_line=second_line
        self.third_line=third_line
        self.last_two=last_two"""
        #set of part of speach
        with open('postag_dict_all.p','rb') as f:
            postag_dict=pickle.load(f)
        self.postag=postag_dict[2]
    def gen_limerick(self, word, templates_dataset=None):
        if type(templates_dataset)==type(None):
            dataset, second_line, third_line, last_two=get_templates_new()#function in functions.py
        #####
        words=self.mp.get_five_words(word)[1:]
        print('Five words are: ', words)
        ###########
        if not self.gen.in_vocab(words):
            print ('Words not in vocab')
            return None
        ########################
        #get postag of 4 words
        postag_words=[]
        for x in words:
            postag_words.append(self.postag[x][0])
        print (postag_words)
        #### get templates
        if type(templates_dataset)==type(None):
            try:
                template_2=random.choice(second_line[postag_words[0]])
                template_3=random.choice(third_line[postag_words[1]])
                template_4=random.choice(dataset[postag_words[2]])
                template_5=random.choice(dataset[postag_words[3]])
            except KeyError:
                print ('POS not in set of templates')
                return None
        else:
            template_2, template_3, template_4, template_5=templates_dataset
        
        
        #######################
        #2nd line############
        if type(template_2)==tuple:
            print(template_2)
            template_2=template_2[0]
        line_2=self.gen.genPoem_backward(words[0],template_2)
        ##################
        #3rd line
        if type(template_3)==tuple:
            print(template_3)
            template_3=template_3[0]
        line_3=self.gen.genPoem_backward(words[1],template_3)
        ###############
        #4th line
        if type(template_4)==tuple:
            print(template_4)
            template_4=template_4[0]
        line_4=self.gen.genPoem_backward(words[2],template_4)
        #############
        #5th line
        if type(template_5)==tuple:
            print(template_5)
            template_5=template_5[0]
        line_5=self.gen.fifth_line(line_4[0][1][1], words[-1], template_5)
        print (template_2)
        print (template_3)
        print (template_4)
        print (template_5)

        print ('*************\n')
        print('\n'+' '.join(line_2[0][1][1]))
        print(' '.join(line_3[0][1][1]))
        print(' '.join(line_4[0][1][1]))
        print(' '.join(line_5[0][1][1][1:]))