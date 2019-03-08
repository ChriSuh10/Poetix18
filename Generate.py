from six.moves import cPickle

import numpy as np
import tensorflow as tf
import pandas as pd

from gensim.models import KeyedVectors
from collections import defaultdict

import random
import time
import os

from model_back import Model as Model_back
from model_forw import Model as Model_forw
from functions import *



from gensim.parsing.preprocessing import remove_stopwords
from nltk.corpus import wordnet as wn
import os
from six.moves import cPickle
import requests

class Generate:
    def __init__(self, wv_file=None, wv=None, save_dir_back="frost_model"):
        #LOAD DIRECTORY OF MODELS
        text_list = [("data/all_combined/input.txt","all_combined_forward"),("data\all_combined\input.txt","all_combined_back")]
        #np.random.shuffle(text_list)
        t = text_list[0][0] #THIS TEXT IS THE VOCAB!
        self.save_dir = text_list[0][1] #directory for forward model
        t_back=text_list[1][0]
        self.save_dir_back=text_list[1][1]#directory for backwards model

        if wv_file is None and wv is None:
            raise ValueError('Must specify workd vectors')

        # load glove to a gensim model
        if wv is not None:
            glove_model = wv
        else:
            glove_model = KeyedVectors.load_word2vec_format(wv_file, binary=False)

        # system arguments
        topics = [sys.argv[1]]
        try:
            seed = int(sys.argv[2])
        except:
            seed = 1

        text = open(t, encoding='latin-1')
        text = text.read()

        np.random.seed(seed) # seed for reproducibility
        words = [simple_clean(word) for word in text.split()]
        uniques = set()
        for word in words:
            uniques.add(word)
        corpus = list(uniques)
        dictWordTransitions = {}
        glove_words = glove_model.vocab.keys() #SHOULDN'T THIS HAVE BEEN MOST COMMON WORDS IN THE CORPUS?
        # CORPUS LIMITED TO WORDS IN GLOVE VOCAB
        word_counts = collections.Counter([x for x in words])
        corpi = [x[0] for x in word_counts.most_common() if x[0] in glove_words]

        # Class variables
        self.dictPartSpeechTags = createPartSpeechTags(corpi)
        self.dictPossiblePartsSpeech = possiblePartsSpeechPaths()
        vowels = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
        self.width = 20
        self.wordPools = [set([]) for n in range(4)]
        with open('postag_dict_all.p', 'rb') as f:
            self.postag_dict = pickle.load(f)

        self.PartOfSpeachSet=self.postag_dict[1]
        self.TemplatePOS=['PRP', 'VBZ', 'DT', 'NN', 'VBG', 'TO', 'DT', 'NN']

    def get_postag_dict(self):
        return self.postag_dict

    def get_save_dir(self):
        return self.save_dir

    def get_save_dir_back(self):
        return self.save_dir_back

    def get_dict_tag(self):
        return self.dictPartSpeechTags

    def get_dict_pos(self):
        return self.dictPossiblePartsSpeech

    def get_width(self):
        return self.width

    def get_wordPools(self):
        return self.wordPools

    def get_pos_set(self):
        return self.PartOfSpeachSet

    def get_template_pos(self):
        return self.TemplatePOS

        
        

#return list of generated lines with their states and scores. Sorted by scores.
#inputs: LAST word in the line, template POS. Optional: initial state and initial score
    def genPoem_backward(self, word_last, TemplatePOS):
        start = time.time()
        #function to sample from the first "cut" lines with highest probs
        #currently not used
        def sampleLine(lst, cut):
            ''' samples from top "cut" lines, the distribution being the softmax of true sentence probabilities'''
            probs = list()
            for i in range(cut):
                probs.append(np.exp(lst[i][0][0][0]))
            probs = np.exp(probs) / sum(np.exp(probs))
            index = np.random.choice(cut,1,p=probs)[0]
            return lst[index][1][1]
        #load barckwards model
        tf.reset_default_graph()
        with open(os.path.join(self.save_dir_back, 'config.pkl'), 'rb') as f:
            saved_args = cPickle.load(f)
        with open(os.path.join(self.save_dir_back, 'words_vocab.pkl'), 'rb') as f:
            word_keys, vocab = cPickle.load(f)
        model = Model_back(saved_args, True)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self.save_dir_back)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                poem = []
                line_num = 0
                wordPool_ind = 0
                #get initial states and scores
                state = sess.run(model.initial_state)
                init_score = np.array([[0]])
                #search barckwards function in functions.py
                lst = search_back_no_rhymes(model, vocab, init_score,[word_last],state, sess, 1,\
                                  self.dictPartSpeechTags,self.dictPossiblePartsSpeech,self.width,self.wordPools[wordPool_ind], self.PartOfSpeachSet, TemplatePOS)
                lst.sort(key=itemgetter(0), reverse = True)
        print("Generation took {:.3f} seconds".format(time.time() - start))
        return lst

#return list of generated lines with their states and scores. Sorted by scores.
#inputs: FIRST word in the line, template POS. Optional: initial state and initial score
    def genPoem_forward(self, word_last, TemplatePOS, init_state=None, init_score=None):
        start = time.time()
        #sample according to score
        def sampleLine(lst, cut):
            ''' samples from top "cut" lines, the distribution being the softmax of true sentence probabilities'''
            probs = list()
            for i in range(cut):
                probs.append(np.exp(lst[i][0][0][0]))
            probs = np.exp(probs) / sum(np.exp(probs))
            index = np.random.choice(cut,1,p=probs)[0]
            return lst[index][1][1]
            #load forward model
        tf.reset_default_graph()
        with open(os.path.join(self.save_dir, 'config.pkl'), 'rb') as f:
            saved_args = cPickle.load(f)
        with open(os.path.join(self.save_dir, 'words_vocab.pkl'), 'rb') as f:
            word_keys, vocab = cPickle.load(f)
        model = Model_forw(saved_args, True)
        print (len(vocab))
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self.save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                poem = []
                line_num = 0
                wordPool_ind = 0
                #if states and scores not specified, initialize them from zero
                if type(init_state)==type(None):
                    state = sess.run(model.initial_state)
                else:
                    state=init_state
                if type(init_score)==type(None):
                    init_score = np.array([[0]])
                #function in functions.py 
                lst = search_forward(model, vocab, init_score,[word_last],state, sess, 1,\
                                  self.dictPartSpeechTags,self.dictPossiblePartsSpeech,self.width,self.wordPools[wordPool_ind], self.PartOfSpeachSet, TemplatePOS)
                lst.sort(key=itemgetter(0), reverse = True)
                # line diagnostics
                #for i in range(10)
                #    print(lst[i][0][0][0], lst[i][1][1])
                #choice = sampleLine(lst, min(10,len(lst)))
                #print (choice)
                #poem.append(choice)
                #line_num+=1
                #if(line_num>3):
                #    line_num = 0
                #    wordPool_ind+=1
            #poem = postProcess(poem,sess,vocab,model)
        print("Generation took {:.3f} seconds".format(time.time() - start))
        return lst
#given a list (such us the output of genPoem_backward or genPoem_forward) get the probs of the next word being "next_word"
#Input: lst (list in the such as it was the same form that the output of genPoem_backward); next_word

    def force_middle(self, lst, next_word):
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
       #load forward model
        tf.reset_default_graph()
        with open(os.path.join(self.save_dir, 'config.pkl'), 'rb') as f:
            saved_args = cPickle.load(f)
        with open(os.path.join(self.save_dir, 'words_vocab.pkl'), 'rb') as f:
            word_keys, vocab = cPickle.load(f)
        model = Model_forw(saved_args, True)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self.save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                poem = []
                line_num = 0
                wordPool_ind = 0
                ret=[]
                for tup in lst:
                    #initialize states and scores
                    state_init = tup[1][0][1]
                    prob_seq =  tup[1][0][0]
                    sequence=tup[1][1]
                    scale = .02 #scale is the significant magnitude required to affect the score of bad/good things
                    dist, state = model.compute_fx(sess, vocab, prob_seq, sequence, state_init, 1)
                    #PREVENTS REPEAT ADJACENT WORDS OR PROBLEM-TAGGED WORDS
                    if(next_word == sequence[-1]):
                        continue
                    #if(partsOfSpeechFilter(sequence[-1],next_word,dictPartSpeechTags,dictPossiblePartsSpeech)):
                    #    continue
                    #FACTORS IN SCORE ADJUSTMENTS
                    score_adjust = decayRepeat(next_word, sequence, 100*scale) #repeats
                    score_adjust += scale*len(next_word)/50 #length word
                    #if(next_word in wordPool):
                    #    score_adjust += scale
                    #CALCULATES ACTUAL SCORE
                    key = np.array([[vocab[next_word]]])
                    new_prob = dist[key]
                    score_tuple = (new_prob, state)
                    score_tup = (score_tuple[0]+score_adjust,score_tuple[1]) #NOTE SHOULD SCORE_ADJUST BE ADDED HERE OR JUST IN THE ITEM LINE?
                    item = (score_tup[0],(score_tup, sequence+[next_word]))
                    if(item[0]==[[-float("inf")]]):
                        continue
                    ret+=[item]
                ret.sort(key=itemgetter(0), reverse = True)
                return ret[:150]

    #function to get a template based on two words
    #currently not used
    def pos_synset(self, words, pos_dict):
        postag_nn=[]
        print(words)
        for word in words:
            pos_word=self.postag_dict[2][word][0]
            postag_nn.append(pos_word)
        print (postag_nn)
        pos=str(postag_nn[0])+'-'+str(postag_nn[1])
        if pos in set(list(pos_dict.keys())):
            possible_templates=pos_dict[pos]
            possible=[k_ for k_ in possible_templates if len(k_[0])<8]
        else:
            pos=str(postag_nn[1])+'-'+str(postag_nn[0])
            if pos in set(list(pos_dict.keys())):
                possible_templates=pos_dict[pos]
                possible=[k_ for k_ in possible_templates if len(k_[0])<8]
                #postag_nn=[postag_nn[1], postag_nn[0]]
                words=[words[1], words[0]]
            else:
                return None
        template_list=list(random.choice(possible))
        if postag_nn[0]==postag_nn[1]:
            template_list.append(words)
            return template_list
        else:
            ww=[words[postag_nn.index(postag_nn[0])], words[postag_nn.index(postag_nn[1])]]
            template_list.append(ww)
            return template_list

    def place_words_in_template(self, words, template):
        postag_nn=[]
        print(words)

        if words[0] not in self.postag_dict[2] or words[1] not in self.postag_dict[2]:
                return None

        w1_pos = self.postag_dict[2][words[0]][0]
        w2_pos = self.postag_dict[2][words[1]][0]

        if w1_pos not in template or w2_pos not in template:
            return None

        positions = []
        for i, pos in enumerate(template):
            if len(positions) >= 2:
                break
            if pos == w1_pos:
                positions.append(i)
                # if already an index, word order needs to be reversed
                if len(positions) >= 2:
                    words.reverse()
                continue
            if pos == w2_pos:
                positions.append(i)

        if len(positions) < 2:
            return None

        template_list=[]
        template_list.append(template)
        template_list.append(positions)
        template_list.append(words)
        return template_list

#given two words generate line.
#postag_: template such as contains the position in which word1 and word2 belong to.
    def generate_line(self, word1, word2, postag_=None):
        nouns=[word1, word2]

        postag_nn=[]
        #SELECT A POS SAMPLE
        if postag_==None:
            postag=self.pos_synset(nouns, self.postag_dict[0])
        else:
            postag=postag_
        if postag==None:
            return None
        last_position=np.argmax(postag[1])
        first_position=np.argmin(postag[1])


        last_noun=postag[2][last_position]
        first_noun=postag[2][first_position]

        first_pos=postag[1][first_position]
        last_pos=postag[1][last_position]
        template=postag[0]

        print ('#################################\n')
        #generate text between word1 and word2
        list_1= self.genPoem_forward(first_noun, template[first_pos:last_pos])
        #get scores of list_1 such as next word is last_noun
        list_2=self.force_middle(list_1, last_noun)
        #load forward model
        tf.reset_default_graph()
        with open(os.path.join(self.save_dir, 'config.pkl'), 'rb') as f:
            saved_args = cPickle.load(f)
        with open(os.path.join(self.save_dir, 'words_vocab.pkl'), 'rb') as f:
            word_keys, vocab = cPickle.load(f)
        model = Model_forw(saved_args, True)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self.save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                poem = []
                line_num = 0
                wordPool_ind = 0
                tt_3=[]
                for element in list_2:
                    #initialize the model from word2
                    seq=element[1][1]
                    state = element[1][0][1]
                    init_score = element[1][0][0]
                    seq=element[1][1]
                    #generate text after word2
                    lst = search_forward(model, vocab, init_score,seq,state, sess, 1,\
                                      self.dictPartSpeechTags,self.dictPossiblePartsSpeech,self.width,self.wordPools[wordPool_ind], self.PartOfSpeachSet, template[first_pos:])
                    tt_3+=lst
                tt_3.sort(key=itemgetter(0), reverse = True)
        #load backwards model
        tf.reset_default_graph()
        with open(os.path.join(self.save_dir_back, 'config.pkl'), 'rb') as f:
            saved_args = cPickle.load(f)
        with open(os.path.join(self.save_dir_back, 'words_vocab.pkl'), 'rb') as f:
            word_keys, vocab = cPickle.load(f)
        model = Model_back(saved_args, True)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self.save_dir_back)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                poem = []
                line_num = 0
                wordPool_ind = 0
                tt_4=[]
                for tup in tt_3[:10]:
                    seq=tup[1][1]
                    #get states of the generated text (from word1 to the end of the line)
                    init_score, state=model.score_a_list(sess, vocab, seq)
                    #generate text before word1
                    lst = search_back_no_rhymes(model, vocab, init_score,seq,state, sess, 1,\
                                  self.dictPartSpeechTags,self.dictPossiblePartsSpeech,self.width,self.wordPools[wordPool_ind], self.PartOfSpeachSet, template)

                    tt_4+=lst
        tt_4.sort(key=itemgetter(0), reverse = True)


        return template, tt_4
        
#similar to generate_line but without manual templates
    def generate_line_2(self, word1, word2, template=None):
        nouns=[word1, word2]

        postag_nn=[]
        #SELECT A POS SAMPLE
        """for noun in nouns:
            postag_nn.append(nltk.pos_tag([noun])[0][1])
        if postag_nn.count('NN')>1:
            postag=list(random.choice(postag_dict[0]['NN-NN']))
            nouns=nouns
            postag.append(nouns)
        elif postag_nn.count('NN')==1 and postag_nn.count('NNS')==1:
            postag=list(random.choice(postag_dict[0]['NN-NNS']))
            nouns=[nouns[postag_nn.index('NN')], nouns[postag_nn.index('NNS')]]
            postag.append(nouns)
        elif postag_nn.count('NNS')>1:
            postag=list(random.choice(postag_dict[0]['NNS-NNS']))
            nouns=nouns
            postag.append(nouns)
        else:
            print ('WORDS ARE NOT NOUNS')
            #return None
            """
        if template is None:
            postag= self.pos_synset(nouns, self.postag_dict[0])
        else:
            postag= self.place_words_in_template(nouns, template)
        if postag==None:
            return None

        last_position=np.argmax(postag[1])
        first_position=np.argmin(postag[1])


        last_noun=postag[2][last_position]
        first_noun=postag[2][first_position]

        first_pos=postag[1][first_position]
        last_pos=postag[1][last_position]
        template=postag[0]
            #postag=[['CC', 'PRP', 'RB', 'NN', 'IN', 'DT', 'NN', 'IN', 'NN'], [3,6], ['night', 'moon']]
        print (postag[0])

        print ('First noun: '+ str(first_noun))
        print ('Last noun: '+ str(last_noun))
        print (template[first_pos:last_pos])

        print ('#################################\n')

        list_1= self.genPoem_forward(first_noun, template[first_pos:last_pos])
        list_2= self.force_middle(list_1[:50], last_noun)


        tf.reset_default_graph()
        with open(os.path.join(self.save_dir, 'config.pkl'), 'rb') as f:
            saved_args = cPickle.load(f)
        with open(os.path.join(self.save_dir, 'words_vocab.pkl'), 'rb') as f:
            word_keys, vocab = cPickle.load(f)
        model = Model_forw(saved_args, True)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self.save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                poem = []
                line_num = 0
                wordPool_ind = 0
                tt_3=[]
                for element in list_2:
                    seq=element[1][1]
                    state = element[1][0][1]
                    init_score = element[1][0][0]
                    seq=element[1][1]
                    lst = search_forward(model, vocab, init_score,seq,state, sess, 1,\
                                      self.dictPartSpeechTags,self.dictPossiblePartsSpeech,self.width,self.wordPools[wordPool_ind], self.PartOfSpeachSet, template[first_pos:])
                    tt_3+=lst
                tt_3.sort(key=itemgetter(0), reverse = True)

        tf.reset_default_graph()
        with open(os.path.join(self.save_dir_back, 'config.pkl'), 'rb') as f:
            saved_args = cPickle.load(f)
        with open(os.path.join(self.save_dir_back, 'words_vocab.pkl'), 'rb') as f:
            word_keys, vocab = cPickle.load(f)
        model = Model_back(saved_args, True)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self.save_dir_back)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                poem = []
                line_num = 0
                wordPool_ind = 0
                tt_4=[]
                for tup in tt_3[:30]:
                    seq=tup[1][1]
                    init_score, state=model.score_a_list(sess, vocab, seq)
                    lst = search_back(model, vocab, init_score,seq,state, sess, 1,\
                                  self.dictPartSpeechTags,self.dictPossiblePartsSpeech,self.width,self.wordPools[wordPool_ind], self.PartOfSpeachSet, template)

                    tt_4+=lst
            else:
                print(ckpt)
        tt_4.sort(key=itemgetter(0), reverse = True)


        return template, tt_4

    def generate_line_collocations(self, word1, word2):
        # check for collocations and fill in
        postag= self.pos_synset([word1, word2], self.postag_dict[0])

        last_position=np.argmax(postag[1])
        first_position=np.argmin(postag[1])


        last_noun=postag[2][last_position]
        first_noun=postag[2][first_position]

        first_pos=postag[1][first_position]
        last_pos=postag[1][last_position]
        template=postag[0]

        line = ['' for i in range(len(template))]
        line[first_position] = first_noun
        line[last_position] = last_noun

        return line



    def print_gen_line(self, word1, word2):
        temp_score=[]
        for x in range(8):
            try:
                template, k = self.generate_line(word1, word2)
            except KeyError:
                continue
            if k!=[]:
                k=k[0]
                #print (k[0][0][0])
                row=(' '.join(k[1][1]),k[0]/len(k[1][1]), template)
                temp_score.append(row)
        return pd.DataFrame(temp_score, columns=['line', 'score', 'POS'])

    def generalization_score(self, word_pair_list, template):
        sum_score = 0
        num_no_fit = 0
        for word1, word2 in word_pair_list:
            try:
                template, k = self.generate_line(word1, word2, template=template)
            except TypeError:
                # print('Words don\'t fit into template')
                num_no_fit += 1
                continue
            if k != []:
                # add score normalized by line length
                sum_score += (k[0][0] / len(k[0][1][1]))
        return sum_score / len(word_pair_list), template, num_no_fit

    def random_pair_list(self, length):
        ret = []
        for i in range(length):
            words = random.sample(self.postag_dict[2].keys(), 2)
            ret.append(words)
        return ret

    def assign_generalization_scores(self, pairs_length):
        new_postag_dict = defaultdict(list)
        num_no_fit = 0
        for k in self.postag_dict[0].keys():
            list_with_scores = []
            for template in self.postag_dict[0][k]:
                score, template, no_fit = self.generalization_score(self.random_pair_list(pairs_length), template)
                list_with_scores.append((score, template))
                num_no_fit += no_fit
            new_postag_dict[k] = list_with_scores
        return new_postag_dict, num_no_fit

    def insert_collocations(self, template, line, collocations):
        for i, w in enumerate(line):
            if w not in collocations:
                continue;

            grams = collocations[w]
            for gram in grams:
                col_word = gram[0][1]

                if len(self.postag_dict[2][col_word]) == 0:
                    continue

                mean_dist = int(round(gram[1]))
                gram_pos = self.postag_dict[2][col_word][0]

                #guard
                if 0 > i + mean_dist or i + mean_dist >= len(line):
                    continue

                # empty spot and pos matches
                if line[i + mean_dist] is '' and template[i + mean_dist] is gram_pos:
                    line[i + mean_dist] = col_word
                    break
        return line

    def generate_line_collocations(self, word1, word2, collocations):
        # check for collocations and fill in
        postag= self.pos_synset([word1, word2], self.postag_dict[0])

        last_position=np.argmax(postag[1])
        first_position=np.argmin(postag[1])

        last_noun=postag[2][last_position]
        first_noun=postag[2][first_position]

        first_pos=postag[1][first_position]
        last_pos=postag[1][last_position]
        template=postag[0]

        line = ['' for i in range(len(template))]
        line[min(postag[1])] = first_noun
        line[max(postag[1])] = last_noun

        line = self.insert_collocations(template, line, collocations)

        return postag, line
        
        
#generate last line for limericks
#fourth_line: list of words of the fourth line
#next_word: last word of 5th line
#template: template for 5th line
    def fifth_line(self, fourth_line, next_word, template):
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
            
        #load forward model
        tf.reset_default_graph()
        with open(os.path.join(self.save_dir, 'config.pkl'), 'rb') as f:
            saved_args = cPickle.load(f)
        with open(os.path.join(self.save_dir, 'words_vocab.pkl'), 'rb') as f:
            word_keys, vocab = cPickle.load(f)
        model = Model_forw(saved_args, True)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self.save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                #initialize state and prob from line 4
                prob, state=model.score_a_list(sess, vocab, fourth_line)
        #get postag for the last word of the fourth line
        last_pos=nltk.pos_tag([fourth_line[-1]])[0][1]
        #generate fifth line conditioned on 4th line
        list_1= self.genPoem_forward(fourth_line[-1], [last_pos]+template[:-1], init_state=state, init_score=prob)
        list_2=self.force_middle(list_1, next_word)
        list_2.sort(key=itemgetter(0), reverse = True)
        return list_2
#check if the words of a list are in the vocab. Useful to check the 5 words.
    def in_vocab(self, lst):
        tf.reset_default_graph()
        with open(os.path.join(self.save_dir, 'config.pkl'), 'rb') as f:
            saved_args = cPickle.load(f)
        with open(os.path.join(self.save_dir, 'words_vocab.pkl'), 'rb') as f:
            word_keys, vocab = cPickle.load(f)
        vocabulary=set(list(vocab.keys()))
        for x in lst:
            if x not in vocabulary:
                return False
        return True
