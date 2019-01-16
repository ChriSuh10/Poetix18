from six.moves import cPickle

import numpy as np
import tensorflow as tf
import pandas as pd
import random
import time
import os

from model_back import Model as Model_back
from model_forw import Model as Model_forw
from functions import *

class Generate:
    def __init__(self):
        #LOAD DIRECTORY OF MODELS
        text_list = [("data/frost/input.txt","save_2"),("data\frost\input.txt","frost_model")]
        #np.random.shuffle(text_list)
        t = text_list[0][0] #THIS TEXT IS THE VOCAB!
        self.save_dir = text_list[0][1] #THIS IS THE MODEL DIRECTORY
        t_back=text_list[1][0]
        self.save_dir_back=text_list[1][1]

        # load glove to a gensim model
        glove_model = KeyedVectors.load_word2vec_format('/Users/chris/Downloads/glove.6B/glove.6B.300d.w2v.txt',binary=False)

        # system arguments
        topics = [sys.argv[1]]
        try:
            seed = int(sys.argv[2])
        except:
            seed = 1

        text = open(t)
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
        with open('postag_dict.p', 'rb') as f:
            self.postag_dict = pickle.load(f)

        self.PartOfSpeachSet=self.postag_dict[1]
        self.TemplatePOS=['PRP', 'VBZ', 'DT', 'NN', 'VBG', 'TO', 'DT', 'NN']

    def get_postag_dict(self):
        return self.postag_dict

    def get_save_dir(self):
        return self.save_dif

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

    def genPoem_forward(self, word_last, TemplatePOS):
        start = time.time()
        def sampleLine(lst, cut):
            ''' samples from top "cut" lines, the distribution being the softmax of true sentence probabilities'''
            probs = list()
            for i in range(cut):
                probs.append(np.exp(lst[i][0][0][0]))
            probs = np.exp(probs) / sum(np.exp(probs))
            index = np.random.choice(cut,1,p=probs)[0]
            return lst[index][1][1]
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
                state = sess.run(model.initial_state)
                init_score = np.array([[0]])
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
                ret=[]
                for tup in lst:
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
                return ret[:self.width]

    #function to get a template based on two words
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

    def generate_line(self, word1, word2):
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
        postag= self.pos_synset(nouns, self.postag_dict[0])
        if postag==None:
            return None
        #postag=[['CC', 'PRP', 'RB', 'NN', 'IN', 'DT', 'NN', 'IN', 'NN'], [3,6], ['night', 'moon']]
        print (postag[0])
        last_position=np.argmax(postag[1])
        first_position=np.argmin(postag[1])


        last_noun=postag[2][last_position]
        first_noun=postag[2][first_position]

        first_pos=postag[1][first_position]
        last_pos=postag[1][last_position]
        template=postag[0]
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
