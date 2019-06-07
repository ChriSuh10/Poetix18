import progressbar
import tensorflow as tf
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords
import collections
import tqdm
import os
import re
import random
import itertools
import requests
import pickle
import heapq
import py_files.Rhetoric as Rhetoric

from .model_back import Model as Model_back
from .functions import search_back_meter
from .templates import get_templates

from gpt2.src.score import score_model
from gpt2.src.generate_prompt import generate_prompt
from gpt2.src.encoder import get_encoder
from .templates import get_first_nnp
import pickle

class Limerick_Generate:

    def __init__(self, wv_file='py_files/saved_objects/poetic_embeddings.300d.txt',
            syllables_file='py_files/saved_objects/cmudict-0.7b.txt',
            postag_file='py_files/saved_objects/postag_dict_all.p',
            model_dir='py_files/models/all_combined_back',
            model_name='117M'):
        self.api_url = 'https://api.datamuse.com/words'
        self.ps = nltk.stem.PorterStemmer()
        self.punct = re.compile(r'[^\w\s]')
        self.model_dir = model_dir
        self.model_name = model_name
        # self.poetic_vectors = KeyedVectors.load_word2vec_format(wv_file, binary=False)

        self.create_syll_dict(syllables_file)

        with open(postag_file, 'rb') as f:
            postag_dict = pickle.load(f)
        self.pos_to_words = postag_dict[1]
        self.words_to_pos = postag_dict[2]
        self.create_pos_syllables()
        self.create_templates_dict(postag_dict[0])

        self.first_line_words=pickle.load(open('py_files/saved_objects/first_line.p','rb'))
        self.width = 20
        # Not sure what this does, necessary for search_back function
        self.word_pools = [set([]) for n in range(4)]
        self.enc = get_encoder(self.model_name )

    def create_syll_dict(self, fname):
        """
        Using the cmudict file, returns a dictionary mapping words to their
        intonations (represented by 1's and 0's). Assumed to be larger than the
        corpus of words used by the model.

        Parameters
        ----------
        fname : str
            The name of the file containing the mapping of words to their
            intonations.
        """
        with open(fname, encoding='UTF-8') as f:
            lines = [line.rstrip("\n").split() for line in f if (";;;" not in line)]
            self.dict_meters = {}
            for i in range(len(lines)):
                line = lines[i]
                newLine = [line[0].lower()]
                if("(" in newLine[0] and ")" in newLine[0]):
                    newLine[0] = newLine[0][:-3]
                chars = ""
                for word in line[1:]:
                    for ch in word:
                        if(ch in "012"):
                            if(ch == "2"):
                                chars+="1"
                            else:
                                chars+=ch
                newLine+=[chars]
                lines[i] = newLine
                if(newLine[0] not in self.dict_meters): #THIS IF STATEMENT ALLOWS FOR MULTIPLE PRONUNCIATIONS OF A WORD
                    self.dict_meters[newLine[0]]=[chars]
                else:
                    if(chars not in self.dict_meters[newLine[0]]):
                        self.dict_meters[newLine[0]]+=[chars]
            self.dict_meters[','] = ['']
            self.dict_meters['.'] = ['']

    def create_pos_syllables(self):
        """
        Creates a mapping from every pos encountered in the corpus to the all of
        the possible number of syllables across all of the words tagged with
        the given pos.
        """
        self.pos_syllables = {}
        for k, v in self.pos_to_words.items():
            self.pos_syllables[k] = set()
            for w in v:
                try:
                    self.pos_syllables[k].add(len(self.dict_meters[w][0]))
                except:
                    continue
        self.pos_syllables[','].add(0)
        self.pos_syllables['.'].add(0)

    def create_templates_dict(self, templates):
        """
        Creates a mapping from every (pos, length of line) encountered in the
        corpus to a list of templates ending with that pos and length.

        Parameters
        ----------
        templates : dict
            A dictionary mapping a pairing of pos to templates containing both
            those pos's (used in previous poem generating algorithms).
        """
        self.templates_dict = {}
        for l in templates.values():
            for t, _ in l:
                if len(t) > 15:
                    continue
                ending_pos = t[-1]
                if (ending_pos, len(t)) not in self.templates_dict:
                    self.templates_dict[(ending_pos, len(t))] = []
                self.templates_dict[(ending_pos, len(t))].append(t)

    def two_word_link(self, w1, w2, seen_words):
        """
        Given two words, returns a third word from the set of words contained in
        the definitions of the given words that is closest in the vector space
        to both of the given words.

        Parameters
        ----------
        w1, w2 : str
            The two words used to find the third word that is close to both of
            them in the self.poetic_vectors vector space.

        Returns
        -------
        str
            The third word close to w1 and w2.
        """
        max_sim = -1
        best_word = None
        best_word_def = None

        word_set = set()

        for synset in wn.synsets(w1):
            clean_def = remove_stopwords(self.punct.sub('', synset.definition()))
            word_set.update(clean_def.lower().split())
        for synset in wn.synsets(w2):
            clean_def = remove_stopwords(self.punct.sub('', synset.definition()))
            word_set.update(clean_def.lower().split())

        for other_word in word_set:
            if other_word not in self.poetic_vectors:
                continue
            sim = self.poetic_vectors.similarity(w1, other_word)
            sim += self.poetic_vectors.similarity(w2, other_word)

            if sim > max_sim and other_word != w1 and other_word != w2 and self.ps.stem(other_word) not in seen_words:
                max_sim = sim
                best_word = other_word

        return best_word

    def get_five_words(self, w2):
        """
        Given a seed word, finds four other words that fit the rhyme scheme of
        a limerick while traversing WordNet in order to find relevant words so
        as to encode a storyline throughout the poem.

        Parameters
        ----------
        w2 : str
            The last word in the second line of the limerick.

        Returns
        -------
        tuple
            A tuple of strings that represent the last word in each of the lines
            of the limerick to be generated.
        """
        w1 = w3 = w5 = None
        seen_words = set([self.ps.stem(w2)])

        # Three connection words
        w_response = requests.get(self.api_url, params={'rel_rhy': w2}).json()
        rhyme_nnp = set(d['word'] for d in w_response).intersection(self.pos_to_words['NNP'])
        # Find a word that rhymes with w2 that is a pronoun
        for r in rhyme_nnp:
            if r in self.words_to_pos and self.ps.stem(r) not in seen_words:
                w1 = r
                seen_words.add(self.ps.stem(w1))
                break

        # Any rhyming word
        for r in w_response:
            if r['word'] in self.words_to_pos and self.ps.stem(r['word']) not in seen_words:
                w5 = r['word']
                seen_words.add(self.ps.stem(w5))
                break

        # Word relating to w2 and w5
        w4 = self.two_word_link(w2, w5, seen_words)
        seen_words.add(self.ps.stem(w4))

        w3_response = requests.get(self.api_url, params={'rel_rhy': w4}).json()

        # Find word most similar to w4 that rhymes with it
        max_sim = 0
        for r in w3_response:
            if r['word'] not in self.words_to_pos or r['word'] not in self.poetic_vectors.vocab:
                continue
            this_sim = self.poetic_vectors.similarity(r['word'], w4)
            if this_sim  > max_sim and self.ps.stem(r['word']) not in seen_words:
                w3 = r['word']
                max_sim = this_sim

        if w5 is None or w3 is None or w1 is None:
            raise ValueError('Cannot generate limerick using ', w2)

        seen_words.add(self.ps.stem(w3))
        return w1, w2, w3, w4, w5

    def valid_permutation_sylls(self, num_sylls, template, last_word_sylls):
        """
        Finds and returns the first integer partition of num_sylls with a total
        number of integers equal to the length of template - 1 for which each
        assignment of syllables to pos is valid.

        Parameters
        ----------
        num_sylls : int
            The total number of syllables to be distributed across the words in
            the line.
        template : list
            A list of str containing the pos for each word in the line.
        last_word_sylls : int
            The number of syllables of the last word in the line.

        Returns
        -------
        list
            A list of ints corresponding to a valid assignment of syllables to
            each word in the line.
        """
        def get_all_partition_size_n(n, partition_size):
            """
            Returns all integer partitions of an int with a partition_size number
            of ints.
            """
            def get_all_partitions(n, I=1):
                yield (n,)
                for i in range(I, n//2 + 1):
                    for p in get_all_partitions(n-i, i):
                        yield (i,) + p
            return [p for p in get_all_partitions(n) if len(p) == partition_size]
        def valid_syll(sylls, template):
            """
            Checks if a template and syllable mapping are compatible.
            """
            for i in range(len(template) - 1):
                # Add in zeros to account for punctuation
                if template[i] == ',' or template[i] == '.':
                    sylls.insert(i, 0)
                if sylls[i] not in self.pos_syllables[template[i]]:
                    return False
            return True
        syllables_left = num_sylls - last_word_sylls
        # Punctuation takes up no syllables, so subtract to get number of partitions
        num_zero_sylls = sum(1 if pos == '.' or pos == ',' else 0 for pos in template)
        num_words_left = len(template) - num_zero_sylls - 1

        for partition in get_all_partition_size_n(syllables_left, num_words_left):
            # Goes through all permutations by index, not numbers,
            # inefficient implementation
            permutations = list(itertools.permutations(partition))
            random.shuffle(permutations)
            for perm in permutations:
                perm = list(perm)
                # Last word is fixed
                perm.append(last_word_sylls)
                if valid_syll(perm, template):
                    return perm
    def gen_line_two_words(self, w1, w2, template=None):
        pass

    def run_gen_model_back(self, seq, template, template_sylls, state=None, score=None):
        """
        Wrapper function to allow easy access to the model for generating new
        lines, or lines conditioned on previously generated lines.

        Parameters
        ----------
        seq : list
            A list of str that represent the currently generated words. This list
            will match up with the template from right to left.
        template : list
            A list of str that have pos encodings for every word that has been
            generated or will be generated in the line.
        template_sylls : list
            A list of ints that represents the number of syllables that each word
            in the line will have.

        Returns
        -------
        array
            A large array containing many things, including the state of the
            model, the score of the line, and the line itself for the top n lines.
            The score can be indexed by [0][0].item, and the line by [0][1][1]
        """
        tf.reset_default_graph()
        with open(os.path.join(self.model_dir, 'config.pkl'), 'rb') as f:
            saved_args = pickle.load(f)
        with open(os.path.join(self.model_dir, 'words_vocab.pkl'), 'rb') as f:
            word_keys, vocab = pickle.load(f)
        model = Model_back(saved_args, True)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                word_pool_ind = 0
                if state is None:
                    state = sess.run(model.initial_state)
                if score is None:
                    score = np.array([[0]])

                # This is where the candidate lines are generated
                lst = search_back_meter(model, vocab, score, seq ,state, sess, 1,
                    self.words_to_pos, self.width, self.word_pools[word_pool_ind],
                    self.pos_to_words, template, template_sylls, self.dict_meters)
                # Sort each candidate line by score
                lst.sort(key=lambda x: x[0], reverse = True)
            else:
                raise IOError('No model checkpoint')
        return lst

    def get_rand_template(self, num_sylls, last_word):
        last_pos = self.words_to_pos[last_word][0]
        last_word_sylls = len(self.dict_meters[last_word][0])
        temp_len = num_sylls - last_word_sylls
        return random.choice(self.templates_dict[(last_pos, temp_len)])
    def compute_next_state(self, state, score, seq):
        tf.reset_default_graph()
        with open(os.path.join(self.model_dir, 'config.pkl'), 'rb') as f:
            saved_args = pickle.load(f)
        with open(os.path.join(self.model_dir, 'words_vocab.pkl'), 'rb') as f:
            word_keys, vocab = pickle.load(f)
        model = Model_back(saved_args, True)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                word_pool_ind = 0
                next_score, next_state=model.compute_fx(sess, vocab, score, seq, state, 1)
        return next_score, next_state
    def gen_first_line(self, w2, num_sylls):
        def get_num_sylls(template):
            n=0
            for x in template:
                n+=len(self.dict_meters[x][0])
            return n

        names=self.first_line_words[0]
        cities=self.first_line_words[1]
        names={x[0]:x[1] for x in names}
        w_response = requests.get(self.api_url, params={'rel_rhy': w2}).json()
        rhyme_names = set(d['word'] for d in w_response).intersection(names.keys())
        rhyme_cities=set(d['word'] for d in w_response).intersection(cities)
        templates=get_first_nnp()
        possible_sentence=[]
        for name in rhyme_names:
            for template in templates[names[name]]:
                if len(self.dict_meters[name][0])+get_num_sylls(template)==num_sylls:
                    possible_sentence.append(template+[name])
        for name in rhyme_cities:
            for template in templates['city']:
                try:
                    if len(self.dict_meters[name][0])+get_num_sylls(template)==num_sylls:
                        possible_sentence.append(template+[name])
                except:
                    continue
        if len(possible_sentence)==0:
            raise ValueError('No lines can be constructed with this metric')
        else:
            return possible_sentence


    def gen_line(self, w1, template=None,num_sylls=10, state=None, score=None):
        """
        Generetes a single line, backwards from the given word, with restrictions
        as given by the provided template, number of syllables in the line.

        Parameters
        ----------
        w1 : str
            The last word in the line, used to generate backwards from.
        template : list, optional
            A list containing pos tags for each word in the line. If None, a
            random template will be sampled from the set of templates ending with
            a pos matching the pos of w1.
        num_sylls : int, optional
            If template is None, then a template that has a length close to the
            number of syllables required will be randomly sampled.

        Returns
        -------
        template : list
            The template that was used to generate the line.
        lst : array
            A large array containing many things, including the state of the
            model, the score of the line, and the line itself for the top n lines.
            The score can be indexed by [0][0].item, and the line by [0][1][1]
        """

        if template is None:
            template = self.get_rand_template(num_sylls, w1)
            # temp_len = random.randint(num_sylls - last_word_sylls - 1, num_sylls - last_word_sylls)
            # template = random.choice(self.templates_dict[(w1_pos, temp_len)])

        print(template)
        # Assign syllables to each pos in template
        last_word_sylls = len(self.dict_meters[w1][0])
        template_sylls = self.valid_permutation_sylls(num_sylls, template, last_word_sylls)

        if template_sylls is None:
            raise ValueError('Cannot construct valid meter using template')

        seq = [w1]
        if state is not None and score is not None:
            lst = self.run_gen_model_back(seq, template, template_sylls, state=state, score=score)
        else:
            lst = self.run_gen_model_back(seq, template, template_sylls)

        return template, lst

    def gen_best_line(self, w1, pos=None, templates=None, set_of_templates=None, rand_templates=5, num_sylls=10, state=None, score=None, return_state=False):
        """
        Generetes a single line by choosing the best of 10 lines whose templates were randomly selected,
        backwards from the given word, with restrictions
        as given by the provided template, number of syllables in the line.

        Parameters
        ----------
        w1 : str
            The last word in the line, used to generate backwards from.
        template : list, optional
            A list containing pos tags for each word in the line. If None, a
            random template will be sampled from the set of templates ending with
            a pos matching the pos of w1.
        num_sylls : int, optional
            If template is None, then a template that has a length close to the
            number of syllables required will be randomly sampled.

        Returns
        -------
        template : list
            The template that was used to generate the line.
        lst : array
            A large array containing many things, including the state of the
            model, the score of the line, and the line itself for the top n lines.
            The score can be indexed by [0][0].item, and the line by [0][1][1]
        """
        if pos is None:
            pos=self.words_to_pos[w1][0]
        if templates is None and set_of_templates is not None:
            try:
                t=set_of_templates[pos]
            except KeyError:
                print('No templates for POS')
                raise ValueError('No lines can be constructed')
            n_templates=min(len(t), rand_templates)
            templates=random.sample(t, k=n_templates)
            #template = self.get_rand_template(num_sylls, w1)
            # temp_len = random.randint(num_sylls - last_word_sylls - 1, num_sylls - last_word_sylls)
            # template = random.choice(self.templates_dict[(w1_pos, temp_len)])

        # Assign syllables to each pos in template
        lines=[]
        for template in templates:
            try:
                t, line=self.gen_line(w1, template=template[0],num_sylls=num_sylls, state=state, score=score)
                this_line = line[0][1][1]
                this_score = line[0][0].item() / len(this_line)
                if return_state:
                    lines.append((this_line, this_score, t, template[1],  line[0][1][0][1]))
                else:
                    lines.append((this_line, this_score, t, template[1]))
            except:
                continue
        lines.sort(key=lambda x: x[1], reverse = True)
        if len(lines)==0:
            raise ValueError('No lines can be constructed')
        return lines


    def gen_poem_independent(self, seed_word, first_line_sylls):
        """
        Takes a seed word and then generates five storyline words to be used as
        the last word of each line. For each line, a template is sampled and
        syllable restrictions are placed, and each line is generated independently
        from the others.

        Parameters
        ----------
        seed_word : str
            The seed word from which the other four words are sourced. This word
            will be used as the last word in the second line.
        first_line_sylls : int
            Sum of syllables contained in the first line. The syllable count for
            every other line is calculated from this value.

        Returns
        -------
        list
            A list of tuples containing (line, score, template), with each index
            of the tuple corresponding to its position within the limerick.
        """
        five_words = self.get_five_words(seed_word)

        lines = []
        third_line_sylls = first_line_sylls - 3
        for i, w in enumerate(five_words):
            # Set number of syllables from generated line dependent on which
            # line is being generated
            if i in [0, 1, 4]:
                this_line_sylls = first_line_sylls
            else:
                this_line_sylls = third_line_sylls

            t, out = self.gen_line(w, num_sylls=this_line_sylls)

            this_line = out[0][1][1]
            this_score = out[0][0].item() / len(this_line)
            lines.append((this_line, this_score, t))
        return lines


    def gen_poem_independent_matias(self, seed_word, first_line_sylls, rand_template=5):
        def get_templates_last(n, key):
            _,_1,_2,data=get_templates()
            df=data[key]
            min_n=min(n,len(df))
            t=random.sample(df, k=min_n)
            fourth=[]
            fifth=[]
            for template in t:
                fourth.append((template[0][:template[2]+1], template[1][:template[2]+1]))
                fifth.append((template[0][template[2]+1:], template[1][template[2]+1:]))
            return fourth, fifth


        five_words = self.get_five_words(seed_word)
        first_line=random.choice(self.gen_first_line(seed_word, first_line_sylls))


        lines = [[first_line]]
        third_line_sylls = first_line_sylls - 4

        dataset, second_line_, third_line_, last_two_lines=get_templates()
        templates=[]
        #try:
        #templates 2nd line:
        #templates.append(random.choice(second_line_[self.words_to_pos[five_words[1]][0]]))
        #templates 3rd line
        #templates.append(random.choice(third_line_[self.words_to_pos[five_words[2]][0]]))
        #templates 4th line
        key=self.words_to_pos[five_words[3]][0]+'-'+self.words_to_pos[five_words[4]][0]
        #temp=random.choice(last_two_lines[key])
        #templates.append((temp[0][:temp[2]+1], temp[1][:temp[2]+1]))
        #templates 5th line
        #templates.append((temp[0][temp[2]+1:], temp[1][temp[2]+1:]))
        #except:
        #    print('POS Not in dataset of templates')
        #    return None
        fourth, fifth=get_templates_last(rand_template, key)
        for i, w in enumerate(five_words):
            # Set number of syllables from generated line dependent on which
            # line is being generated
            if i==0:
                continue
            elif i==1:
                this_line_sylls = first_line_sylls
                out = self.gen_best_line(w, num_sylls=this_line_sylls, set_of_templates=second_line_)
            elif i==2:
                this_line_sylls = third_line_sylls
                out = self.gen_best_line(w, num_sylls=this_line_sylls, set_of_templates=third_line_)
            elif i==3:
                out = self.gen_best_line(w, num_sylls=third_line_sylls, templates=fourth)
            elif i==4:
                 out = self.gen_best_line(w, num_sylls=first_line_sylls, templates=fifth)
            if out is None or out==[]:
                raise ValueError
            print (out)
            lines.append(out[0])
        print("************")
        string=''
        for x in lines:
            string+=' '.join(x[0])+'\n'
        print(string)
        return lines

    def gen_poem_conditioned(self, seed_word, second_line_sylls, rand_template=5):
        five_words = self.get_five_words(seed_word)
        print('five words are: ')
        print(five_words)
        def get_templates_last(n, key):
            _,_1,_2,data=get_templates()
            df=data[key]
            min_n=min(n,len(df))
            t=random.sample(df, k=min_n)
            fourth=[]
            fifth=[]
            for template in t:
                fourth.append((template[0][:template[2]+1], template[1][:template[2]+1]))
                fifth.append((template[0][template[2]+1:], template[1][template[2]+1:]))
            return fourth, fifth

        dataset, second_line_, third_line_, last_two_lines=get_templates()
        #t_2=random.choice(second_line_[self.words_to_pos[five_words[1]][0]])[0]
        key=self.words_to_pos[five_words[3]][0]+'-'+self.words_to_pos[five_words[4]][0]
        fourth, fifth=get_templates_last(rand_template, key)

        #t=random.choice(last_two_lines[key])
        #t_4=t[0][:t[2]+1]
        #t_5=t[0][t[2]+1:]
        #t_1 = random.choice(dataset[self.words_to_pos[five_words[0]][0]])[0]
        #t_3 = random.choice(third_line_[self.words_to_pos[five_words[2]][0]])[0]




        o2 = self.gen_best_line(five_words[1],num_sylls=second_line_sylls, set_of_templates=second_line_)
        line2 = o2[0][0]
        score2 = o2[0][1]
        #state2 = o2[0][1][0][1]
        o3 = self.gen_best_line(five_words[2],num_sylls=second_line_sylls - 3, set_of_templates=third_line_)
        line3 = o3[0][0]
        score3 = o3[0][0]
        last=[]
        for line_4, line_5 in zip(fourth, fifth):
            o5 = self.gen_best_line(five_words[4], num_sylls=second_line_sylls, templates=[line_5], return_state=True)
            line5 = o5[0][0]
            score5 = o5[0][1]#/ len(line5)
            state5=o5[0][-1]
            score_for4, state_for4=self.compute_next_state(state5, score5, line5)
            o4 = self.gen_best_line(five_words[3], num_sylls=second_line_sylls-3, templates=[line_4], state=state_for4, score=score_for4)
            try:
                last.append((o4[0], (line5, score5, line_5[2]), o4[1]))
            except:
                continue
        last.sort(key=lambda x: x[2], reverse = True)
        if len(last)==0:
            raise ValueError('no lines can be constructed')
        line4=last[0][0][0]
        score4=last[0][0][1]
        line5=last[0][1][0]
        score5=last[0][1][1]
        #score_for4, state_for4=self.compute_next_state(state5, score5, line5)
        #o1 = self.run_gen_model_back(line2, t1, second_line_sylls, state=state2, score=score2)
        #t1, o1=self.gen_line(five_words[0], t_1,num_sylls=second_line_sylls, state=state_for1, score=score_for1)


        #line1 = o1[0][1][1]
        #score1 = o1[0][0].item() / len(line1)

        #t4, o4=self.gen_line(five_words[3], t_4,num_sylls=second_line_sylls-3, state=state_for4, score=score_for4)
        #o3 = self.run_gen_line(line4, t3, second_line_sylls - 3, state=state4, score=score4)
        #line4 = o4[0][1][1]
        #score4 = o4[0][0].item() / len(line4)

        lines = []
        for i in range(2, 6):
            num_as_str = str(i)
            this_line = (locals()['line' + num_as_str], locals()['score' + num_as_str], locals()['t' + num_as_str])
            lines.append(this_line)
        print("************")
        string=''
        for x in lines:
            string+=' '.join(x[0])+'\n'
        print(string)
        return lines



    def print_poem(self, seed_word, gen_func, *args):
        """
        Simple utility function to print out the generated poem as well along
        with its score and template.

        Parameters
        ----------
        seed_word : str
            The seed word to be used in generating the limerick.
        gen_func : function
            The function to be used in generating the limerick.
        *args
            The parameters to be passed to the generation funciton, not including
            seed_word, which will be automatically passed.
        """
        gen = gen_func(seed_word, *args)
        print('')
        for line, score, template in gen:
            print('{:60} line score: {:2.3f}'.format(' '.join(line), score))
            print(template)

    def gen_line_gpt_multinomial(self, w, default_template=None, rhyme=False):
        """
        Uses GPT to generate a line given the template restriction and initial sequence
        as given by the provided template, number of syllables in the line.
        Parameters. Sample the most likely next sentence.
        ----------
        w : str
            Initial sequence to start generation. Has to end with a period/comma, etc.
        template : list, optional
            A list containing pos tags for each word in the line. If None, a
            random template will be sampled from the set of templates.
        Returns
        -------
        new_line : array
            The line generated by GPT that satisfies the template POS restrictions
        """

        # Randomly sample template from the dataset
        if default_template:
            template = default_template
        else:
            dataset = get_templates()[2]
            s = sum([len(dataset[key]) for key in dataset.keys()])
            key = np.random.choice(list(dataset.keys()), 1, p=[len(dataset[key])/s for key in dataset.keys()])
            template = dataset[key[0]][random.randint(0, len(dataset[key[0]]))][0]

        new_line = []
        new_line_tokens = []
        for e in w.lower().split():
            new_line_tokens.append(self.enc.encode(e)[0])
        w_response = requests.get(self.api_url, params={'rel_rhy': rhyme}).json()
        rhyme_set = set(d['word'] for d in w_response)
        for i in range(len(template)):
            # Logits is the output of GPT model, encoder is used to decode the output
            logits = score_model(model_name=self.model_name, context_token = [new_line_tokens])
            POS = template[i]
            probability = []
            words = []
            tokens = []
            for index in reversed(np.argsort(logits[0])):
                word = self.enc.decode([index]).lower().strip()
                # Restrict the word to have the POS of the template
                if POS in self.words_to_pos[word.lower().strip()]:
                    # Enforce rhyme if last word
                    if i == len(template) - 1 and rhyme and (word.lower().strip() not in rhyme_set):
                        continue
                    probability.append(logits[0][index])
                    words.append(word)
                    tokens.append(index)

            # Draw from the possible words
            with tf.Session(graph=tf.Graph()) as sess:
                logits = tf.placeholder(tf.double, shape=(1, len(probability)))
                samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
                out = sess.run(samples, feed_dict={
                    logits: [probability]
                })
            new_line_tokens.append(tokens[out[0][0]])
            new_line.append(words[out[0][0]])

        return new_line

    def gen_line_gpt(self, w=None, encodes=None, default_template=None,
        rhyme_word=None, rhyme_set = None, search_space=100, num_sylls=0):
        """
        Uses GPT to generate a line given the template restriction and initial sequence
        as given by the provided template, number of syllables in the line.

        Parameters
        ----------
        w : str
            Initial sequence to start generation. Has to end with a period/comma, etc.
        encodes : str
            Initial encoded sequence to start generation. Has to end with a period/comma, etc.
        default_template : list, optional
            Default grammar template to generate the sentence
        rhyme_word : str, optional
            If a rhyme word is passed in, the sentence generated will rhyme with this word
        rhyme_set : set, optional
            If a rhyme set is passed in, the sentence generated will end with a word in this set

        Returns
        -------
        new_line : array
            The line generated by GPT that satisfies the template POS restrictions
        """

        # Randomly sample template from the dataset
        if default_template:
            template = default_template
        else:
            dataset = get_templates()[2]
            s = sum([len(dataset[key]) for key in dataset.keys()])
            key = np.random.choice(list(dataset.keys()), 1, p=[len(dataset[key])/s for key in dataset.keys()])
            template = dataset[key[0]][random.randint(0, len(dataset[key[0]]))][0]

        new_line = []

        if not rhyme_set and rhyme_word:
            w_response = requests.get(self.api_url, params={'rel_rhy': rhyme_word}).json()
            rhyme_set = set(d['word'] for d in w_response)
            # Include the word itself in the rhyme set
            rhyme_set.add(rhyme)

        # Tuple format: original word array, encode array, log probability of this sentence
        if w:
            sentences = [(w.lower().split(), [], 0)]
            for e in w.lower().split():
                sentences[0][1].append(self.enc.encode(e)[0])
        if encodes:
            sentences = [([], encodes, 0)]

        sylls = None
        if num_sylls > 0:
            if rhyme_word:
                last_word_sylls = len(self.dict_meters[rhyme_word][0])
            else:
                # Set to 2 for test purposes
                last_word_sylls = 2
            sylls = self.valid_permutation_sylls(num_sylls, template, last_word_sylls)
            print(sylls)

        for i in range(len(template)):
            # Logits is the output of GPT model, encoder is used to decode the output
            logits = score_model(model_name=self.model_name, context_token = [s[1] for s in sentences])
            POS = template[i]

            new_sentences = []
            # For each sentence, calculate probability of a new word
            for j in range(len(sentences)):
                # There might be duplicate words such as "And" and " and" and we only need one
                for index in range(len(logits[j])):
                    word = self.enc.decode([index]).lower().strip()
                    if len(word.lower().strip()) == 0:
                        fit_pos = False
                    elif len(self.words_to_pos[word.lower().strip()]) == 0:
                        word_pos = nltk.pos_tag([word.lower().strip()])
                        if len(word_pos) == 0:
                            print(word)
                            fit_pos = False
                        else:
                            fit_pos = POS == word_pos[0][1]
                    else:
                        fit_pos = POS in self.words_to_pos[word]

                    # Restrict the word to have the POS of the template
                    if fit_pos:
                        stripped_word = word.lower().strip()

                        # Enforce rhyme if last word
                        if i == len(template) - 1 and rhyme_set and (stripped_word not in rhyme_set):
                            continue
                        # Enforce meter if meter is provided
                        if sylls and (stripped_word not in self.dict_meters or len(self.dict_meters[stripped_word][0]) != sylls[i]):
                            continue
                        # Add candidate sentence to new array
                        new_sentences.append(
                            (sentences[j][0] + [word],
                            sentences[j][1] + [index],
                            sentences[j][2] + np.log(logits[j][index])))

            # Get the most probable N sentences by sorting the list according to probability
            sentences = heapq.nsmallest(min(len(new_sentences), search_space), new_sentences, key=lambda x: -x[2])
        return sentences[0]

    def gen_line_gpt_rep_multiline(self, figure_info, search_space=100,top_sent=10):

        prompts = figure_info.get_orig_text()
        rep_templates = figure_info.get_repitition_template()
        pos_templates = figure_info.get_pos_template()
        orig_tokens = figure_info.get_orig_tokens()
        banned_set = figure_info.get_orig_rep_words()
        print(banned_set)
        used_words_dict = []

        prompt1 = self.enc.encode(' '.join(prompts[0]))
        rep_template1 = rep_templates[0]
        pos_template1 = pos_templates[0]
        orig_tokens1 = orig_tokens[0]
        gen = self.gen_line_gpt_rep(figure_id=figure_info.get_unique_id(), rep_template=rep_template1, default_template=pos_template1, encodes=prompt1, orig_tokens=orig_tokens1, search_space=search_space, banned_set=banned_set,top_sent=top_sent)
        sent_done = [[' '.join(gen['sents'][j])] for j in range(len(gen['sents']))]

        if(figure_info.get_num_lines() > 1):
            for l in range(0,len(gen['sents'])):
                sent = gen['sents'][l]
                used_words_dict_i = {i: sent[gen['used_words_dict'][i]] for i in gen['used_words_dict'].keys()}
                prompt2 = self.enc.encode(' '.join(prompts[1]))
                rep_template2 = rep_templates[1]
                pos_template2 = pos_templates[1]
                orig_tokens2 = orig_tokens[1]
                gen2 = self.gen_line_gpt_rep(figure_id=figure_info.get_unique_id(), rep_template=rep_template2, default_template=pos_template2, encodes=prompt2, orig_tokens=orig_tokens2,
                                             search_space=top_sent, banned_set=banned_set, used_words_dict=used_words_dict_i, top_sent=1)
                print(gen2)
                sent_done[l].append(' '.join(gen2['sents'][0]))
                print(sent_done)

        print(sent_done)
        return sent_done


    def gen_line_gpt_rep(self, figure_id, rep_template, w=None, encodes=None, orig_tokens=None, default_template=None, rhyme_word=None, rhyme_set=None, banned_set=None, used_words_dict=None, search_space=100,top_sent=10):
        """
        Uses GPT to generate a line given the template restriction and initial sequence
        as given by the provided template, number of syllables in the line.

        Parameters
        ----------
        w : str
            Initial sequence to start generation. Has to end with a period/comma, etc.
        encodes : str
            Initial encoded sequence to start generation. Has to end with a period/comma, etc.
        default_template : list, optional
            Default grammar template to generate the sentence
        rhyme_word : str, optional
            If a rhyme word is passed in, the sentence generated will rhyme with this word
        rhyme_set : set, optional
            If a rhyme set is passed in, the sentence generated will end with a word in this set

        Returns
        -------
        new_line : array
            The line generated by GPT that satisfies the template POS restrictions
        """

        # Randomly sample template from the dataset
        if default_template:
            template = default_template
        else:
            dataset = get_templates()[2]
            s = sum([len(dataset[key]) for key in dataset.keys()])
            key = np.random.choice(list(dataset.keys()), 1, p=[len(dataset[key]) / s for key in dataset.keys()])
            template = dataset[key[0]][random.randint(0, len(dataset[key[0]]))][0]

        new_line = []

        if not rhyme_set and rhyme_word:
            w_response = requests.get(self.api_url, params={'rel_rhy': rhyme_word}).json()
            rhyme_set = set(d['word'] for d in w_response)
            # Include the word itself in the rhyme set
            rhyme_set.add(rhyme)

        # Tuple format: original word array, encode array, log probability of this sentence
        if w:
            sentences = [(w.lower().split(), [], 0)]
            for e in w.lower().split():
                sentences[0][1].append(self.enc.encode(e)[0])
        if encodes:
            sentences = [([], encodes, 0)]

        cc_lookup = dict()
        
        bar = progressbar.ProgressBar(maxval=len(template), \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start() 

        for i in range(len(template)):
            # Logits is the output of GPT model, encoder is used to decode the output
            logits = score_model(model_name=self.model_name, context_token=[s[1] for s in sentences])
            print("Unique id: %d progress ->" % (figure_id))
            bar.update(i+1)
            POS = template[i]
            CC = str(rep_template[i])
            CC_WORD_ID = int(CC[:1])
            CC_INSTANCE_ID = CC_WORD_ID

            if (CC_WORD_ID != 0):
                CC_INSTANCE_ID = int(CC[1:])
                if (CC_INSTANCE_ID == 1):
                    cc_lookup[CC_WORD_ID] = i

            new_sentences = []
            # For each sentence, calculate probability of a new word
            for j in range(len(sentences)):
                # There might be duplicate words such as "And" and " and" and we only need one
                for index in range(len(logits[j])):
                    word = self.enc.decode([index]).lower().strip()
                    CC_MET = False
                    if (CC_INSTANCE_ID == 0):
                        used_set = set(sentences[j][0][d].lower().strip() for d in cc_lookup.values())
                        if(used_words_dict):
                            [used_set.add(used) for used in used_words_dict.values()]
                        if (len(used_set) == 0):
                            CC_MET = True
                        else:
                            if (word not in used_set):
                                CC_MET = True
                            else:
                                CC_MET = False
                    elif (CC_INSTANCE_ID == 1 and CC_WORD_ID > 1):
                        used_set = set(sentences[j][0][d].lower().strip() for d in range(len(sentences[j][0])))
                        if(used_words_dict):
                            [used_set.add(used) for used in used_words_dict.values()]
                        if (len(used_set) == 0):
                            CC_MET = True
                        else:
                            if (word not in used_set):
                                CC_MET = True
                            else:
                                CC_MET = False
                    elif (CC_INSTANCE_ID > 1):
                        # for k, v in cc_lookup.items():
                        # print(k, v)
                        # print(CC)
                        # print(CC_WORD_ID)
                        # print(CC_INSTANCE_ID)
                        #CC_IDX = cc_lookup[CC_WORD_ID]
                        # print(word)
                        # print(sentences[j][0][CC_IDX])

                        cword = ""

                        if(used_words_dict):
                            cword = used_words_dict[CC_WORD_ID]
                        else:
                            CC_IDX = cc_lookup[CC_WORD_ID]
                            cword = sentences[j][0][CC_IDX].lower().strip()


                        if (word.lower().strip() == cword):
                            CC_MET = True
                        # print('True')
                        else:
                            CC_MET = False
                    else:
                        CC_MET = True

                    if (word in banned_set):
                        CC_MET = False

                    if (CC_MET):
                        # Restrict the word to have the POS of the template
                        if POS in self.words_to_pos[word]:
                            # Enforce rhyme if last word
                            if i == len(template) - 1 and rhyme_set and (word.lower().strip() not in rhyme_set):
                                continue
                            # Add candidate sentence to new array
                            new_sentences.append(
                                (sentences[j][0] + [word],
                                 sentences[j][1] + [index],
                                 sentences[j][2] + np.log(logits[j][index])))

                if not new_sentences:
                    for j in range(len(sentences)):
                        new_sentences.append(
                            (sentences[j][0] + [orig_tokens[i]],
                             sentences[j][1] + [-1],
                             sentences[j][2] + 0))

            # Get the most probable N sentences by sorting the list according to
            # probability

            #new_sentences_nd_set = set(tuple(x) for x in new_sentences)
            #new_sentneces_nd = [ list(x) for x in new_sentences_nd_set ]
            #new_sentences_nd.sort(key = lambda x: new_sentences.index(x) )
            #print(new_sentences_nd)

            if(len(new_sentences) < search_space):
                sentences = heapq.nsmallest(len(new_sentences), new_sentences, key=lambda x: -x[2])
            else:
                sentences = heapq.nsmallest(search_space, new_sentences, key=lambda x: -x[2])

            #new_sentences_nd_set = set(tuple(sentences[j][0]) for j in range(len(sentences)))
            #print(new_sentences_nd_set)
            #new_sentneces_nd = [ list(x) for x in new_sentences_nd_set ]
            #print(new_setnences_nd)
            #new_sentences_nd.sort(key = lambda x: sentences.index(x) )
            #sentences = new_sentences_nd
            #sentences = new_sentences_nd
            new_sentences_nd = []
            sent_sent_list = list()
            for j in range(len(sentences)):
                if sentences[j][0] not in sent_sent_list:
                    sent_sent_list.append(sentences[j][0])
                    new_sentences_nd.append(
                        (sentences[j][0],
                         sentences[j][1],
                         sentences[j][2]))
            sentences = new_sentences_nd
            sent_done = [[' '.join(sentences[j][0])] for j in range(top_sent)]
            print(sent_done)

        gen = {'sents': [sentences[j][0] for j in range(top_sent)], 'used_words_dict': cc_lookup}
        return gen

    def gen_line_gpt_cc(self, cctemplate, w=None, encodes=None, default_template=None, rhyme_word=None, rhyme_set=None,
                        banned_set=None, search_space=100):
        """
        Uses GPT to generate a line given the template restriction and initial sequence
        as given by the provided template, number of syllables in the line.

        Parameters
        ----------
        w : str
            Initial sequence to start generation. Has to end with a period/comma, etc.
        encodes : str
            Initial encoded sequence to start generation. Has to end with a period/comma, etc.
        default_template : list, optional
            Default grammar template to generate the sentence
        rhyme_word : str, optional
            If a rhyme word is passed in, the sentence generated will rhyme with this word
        rhyme_set : set, optional
            If a rhyme set is passed in, the sentence generated will end with a word in this set

        Returns
        -------
        new_line : array
            The line generated by GPT that satisfies the template POS restrictions
        """

        # Randomly sample template from the dataset
        if default_template:
            template = default_template
        else:
            dataset = get_templates()[2]
            s = sum([len(dataset[key]) for key in dataset.keys()])
            key = np.random.choice(list(dataset.keys()), 1, p=[len(dataset[key]) / s for key in dataset.keys()])
            template = dataset[key[0]][random.randint(0, len(dataset[key[0]]))][0]

        new_line = []

        if not rhyme_set and rhyme_word:
            w_response = requests.get(self.api_url, params={'rel_rhy': rhyme_word}).json()
            rhyme_set = set(d['word'] for d in w_response)
            # Include the word itself in the rhyme set
            rhyme_set.add(rhyme)

        # Tuple format: original word array, encode array, log probability of this sentence
        if w:
            sentences = [(w.lower().split(), [], 0)]
            for e in w.lower().split():
                sentences[0][1].append(self.enc.encode(e)[0])
        if encodes:
            sentences = [([], encodes, 0)]

        cc_lookup = dict()

        for i in range(len(template)):
            # Logits is the output of GPT model, encoder is used to decode the output
            logits = score_model(model_name=self.model_name, context_token=[s[1] for s in sentences])
            POS = template[i]
            CC = str(cctemplate[i])
            CC_WORD_ID = int(CC[:1])
            CC_INSTANCE_ID = CC_WORD_ID

            if (CC_WORD_ID != 0):
                CC_INSTANCE_ID = int(CC[1:])
                if (CC_INSTANCE_ID == 1):
                    cc_lookup[CC_WORD_ID] = i

            new_sentences = []
            # For each sentence, calculate probability of a new word
            for j in range(len(sentences)):
                # There might be duplicate words such as "And" and " and" and we only need one
                for index in range(len(logits[j])):
                    word = self.enc.decode([index]).lower().strip()
                    CC_MET = False
                    if (CC_INSTANCE_ID == 0):
                        used_set = set(sentences[j][0][d].lower().strip() for d in cc_lookup.values())
                        if (len(used_set) == 0):
                            CC_MET = True
                        else:
                            if (word not in used_set):
                                CC_MET = True
                            else:
                                CC_MET = False
                    elif (CC_INSTANCE_ID == 1 and CC_WORD_ID > 1):
                        used_set = set(sentences[j][0][d].lower().strip() for d in range(len(sentences[j][0])))
                        if (len(used_set) == 0):
                            CC_MET = True
                        else:
                            if (word not in used_set):
                                CC_MET = True
                            else:
                                CC_MET = False
                    elif (CC_INSTANCE_ID > 1):
                        # for k, v in cc_lookup.items():
                        # print(k, v)
                        # print(CC)
                        # print(CC_WORD_ID)
                        # print(CC_INSTANCE_ID)
                        CC_IDX = cc_lookup[CC_WORD_ID]
                        # print(word)
                        # print(sentences[j][0][CC_IDX])
                        if (word.lower().strip() == sentences[j][0][CC_IDX].lower().strip()):
                            CC_MET = True
                        # print('True')
                        else:
                            CC_MET = False
                    else:
                        CC_MET = True

                    if (word in banned_set):
                        CC_MET = False

                    if (CC_MET):
                        # Restrict the word to have the POS of the template
                        if POS in self.words_to_pos[word]:
                            # Enforce rhyme if last word
                            if i == len(template) - 1 and rhyme_set and (word.lower().strip() not in rhyme_set):
                                continue
                            # Add candidate sentence to new array
                            new_sentences.append(
                                (sentences[j][0] + [word],
                                 sentences[j][1] + [index],
                                 sentences[j][2] + np.log(logits[j][index])))

            # Get the most probable N sentences by sorting the list according to probability
            sentences = heapq.nsmallest(min(len(new_sentences), search_space), new_sentences, key=lambda x: -x[2])
        print(new_sentences)
        print(sentences)
        print(sentences[0][0])
        return sentences




    def gen_poem_gpt(self, rhyme1, rhyme2, default_templates, first_line_sylls,
    story_line=False, prompt_length=100, save_as_pickle=False, search_space=100,
    num_sylls = 0):
        """
        Uses GPT to generate a line given the template restriction and initial sequence
        as given by the provided template, number of syllables in the line.

        Parameters
        ----------
        story_line: bool
            Whether to generate five words with a story line and use those as the last words.
        rhyme1 : str
            Initial word to start generation, and the first rhyming word
        rhyme2 : str, optional
            The second word that the third and forth lines have to rhyme with.
            If storyline is set to False this word is necessary.
        first_line_sylls: int
            Number of the syllables that the first line needs to have.
        prompt_length: int
            The length of the prompt that is generated before generating the poem.
            This will influence memory used and should not be too big.
        default_templates : list
            Default grammar templates that the poem uses
        save_as_pickle : bool, optional
            Whether to save the generated prompt and the first line in a file.
            This saves the parameters to the disk enables genrating poems in multiple runs.
        search_space : int, optional
            Search space of the sentence finding algorithm.
            The larger the search space, the more sentences the network runs
            in parallel to find the best one with the highest score.

        Returns
        -------
        None
        """
        if story_line:
            # five_words = self.get_five_words(rhyme1)
            five_words = ('joan', 'loan', 'glue', 'tissue', 'bone')
        else:
            # Get the rhyme sets
            w1_response = requests.get(self.api_url, params={'rel_rhy': rhyme1}).json()
            w2_response = requests.get(self.api_url, params={'rel_rhy': rhyme2}).json()
            r1_set = set(d['word'] for d in w1_response)
            r2_set = set(d['word'] for d in w2_response)

            # Include the word itself in the rhyme set
            r1_set.add(rhyme1)
            r2_set.add(rhyme2)

        # Used the old method to generate the first line
        first_line = random.choice(self.gen_first_line(rhyme1, first_line_sylls))
        print(first_line)
        first_line_encodes = self.enc.encode(" ".join(first_line))
        out = generate_prompt(model_name=self.model_name, seed_word=rhyme1, length=prompt_length)
        prompt = self.enc.decode(out[0][0])
        prompt = prompt[:prompt.rfind(".")+1]
        prompt = self.enc.encode(prompt) + first_line_encodes

        if not story_line:
            r1_set.discard(first_line[-1])

        # Option to save the prompt in a file and generate sentences in different runs
        if save_as_pickle:
            with open('gpt2.pkl', 'wb') as f:
                pickle.dump(prompt, f)
            return

        # Search space is set to decay because the more sentences we run, the longer the prompt
        search_space_coef = [1, 1, 0.5, 0.25]

        for i in range(4):
            curr_sylls = num_sylls - 3 if (i == 2 or i == 3) else num_sylls

            if not story_line:
                rhyme_set = r1_set if (i == 0 or i == 3) else r2_set
                new_sentence = self.gen_line_gpt(w=None, encodes=prompt, default_template = default_templates[i], rhyme_set = rhyme_set, search_space = int(search_space * search_space_coef[i]), num_sylls = curr_sylls)
                rhyme_set.discard(new_sentence[0][-1])
            else:
                new_sentence = self.gen_line_gpt(w=None, encodes=prompt, default_template = default_templates[i], rhyme_set = [five_words[i+1]], search_space = int(search_space * search_space_coef[i]), num_sylls = curr_sylls)

            prompt += new_sentence[1]

    def gen_line_with_template(self, prompt, template, num):
        """
        Uses GPT to generate a line given the template restriction and initial sequence
        as given by the provided template, number of syllables in the line.

        Parameters
        ----------
        w : str
            Initial sequence to start generation. Has to end with a period/comma, etc.
        template : list, optional
            A list containing pos tags for each word in the line. If None, a
            random template will be sampled from the set of templates.

        Returns
        -------
        new_line : array
            The line generated by GPT that satisfies the template POS restrictions
        """
        word_dict = collections.defaultdict(set)

        pos_length={}
        for i in self.pos_to_words.keys():
            pos_length[i]=len(self.pos_to_words[i])

        words = re.sub("[^\w]", " ",  prompt).split()
        for word in words:
            for POS in self.words_to_pos[word.lower()]:
                word_dict[POS].add(word.lower())
        for POS in word_dict.keys():
            word_dict[POS] = list(word_dict[POS])

        results = []; encodes = []

        sentences = [['he', 'would', 'go','to', 'a', 'party'],['i','can','stare','at','the','sky']]
        for i in sentences:
            temp=[self.enc.encode(word)[0] for word in i]
            encodes.append(temp)

        for i in range(num):
            sentence=[]
            for POS in template:
                if pos_length[POS]<=50:
                    w=random.choice(self.pos_to_words[POS])
                else:
                    w=random.choice(word_dict[POS])
                sentence.append(w)
            sentences.append(sentence)
            encodes.append([self.enc.encode(word)[0] for word in sentence])
        encodes=np.array(encodes)

        probs=np.zeros(len(sentences))

        for j in tqdm.trange(1, len(sentences[0])):
            results = score_model(model_name=self.model_name, context_token = encodes[:,:j])
            for i in range(len(sentences)):
                probs[i] +=  np.log(results[i][encodes[i][j]])

        index=np.argsort(np.negative(probs))
        for i in index:
            print("{}:{}".format(probs[i],sentences[i]))

        #print(sentences)
        #print(probs)
        return
