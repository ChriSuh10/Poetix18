import tensorflow as tf
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords

import os
import re
import random
import itertools
import requests
import pickle

from .model_back import Model as Model_back
from .functions import search_back

class Limerick_Generate:

    def __init__(self, wv_file='py_files/saved_objects/poetic_embeddings.300d.txt',
            syllables_file='py_files/saved_objects/cmudict-0.7b.txt',
            postag_file='py_files/saved_objects/postag_dict_all.p',
            model_dir='py_files/models/all_combined_back'):
        self.api_url = 'https://api.datamuse.com/words'
        self.ps = nltk.stem.PorterStemmer()
        self.punct = re.compile(r'[^\w\s]')
        self.model_dir = model_dir
        self.poetic_vectors = KeyedVectors.load_word2vec_format(wv_file, binary=False)
        self.create_syll_dict(syllables_file)

        with open(postag_file, 'rb') as f:
            postag_dict = pickle.load(f)
        self.pos_to_words = postag_dict[1]
        self.words_to_pos = postag_dict[2]
        self.create_pos_syllables()
        self.create_templates_dict(postag_dict[0])

        self.width = 20
        # Not sure what this does, necessary for search_back function
        self.word_pools = [set([]) for n in range(4)]

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
        with open(fname) as f:
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
        Creates a mapping from eery pos encountered in the corpus to a list of
        templates ending with that pos.

        Parameters
        ----------
        templates : dict
            A dictionary mapping a pairing of pos to templates containing both
            those pos's (used in previous poem generating algorithms).
        """
        self.templates_dict = {}
        for l in templates.values():
            for t, _ in l:
                ending_pos = t[-1]
                if ending_pos not in self.templates_dict:
                    self.templates_dict[ending_pos] = []
                self.templates_dict[ending_pos].append(t)

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
            if r['word'] not in self.words_to_pos:
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

    def gen_line(self, w1, template=None, num_sylls=10):
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
        last_word_sylls = len(self.dict_meters[w1][0])
        if template is None:
            w1_pos = self.words_to_pos[w1][0]
            template = []
            # Assumes template of length temp_len exists
            while len(template) <= num_sylls - 2 or len(template) >= num_sylls - last_word_sylls + 1:
                template = np.random.choice(self.templates_dict[w1_pos], 1).item()

        print(template)
        # Assign syllables to each pos in template
        template_sylls = self.valid_permutation_sylls(num_sylls, template, last_word_sylls)

        if template_sylls is None:
            raise ValueError('Cannot construct valid meter using template')

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
                poem = []
                line_num = 0
                word_pool_ind = 0
                state = sess.run(model.initial_state)
                init_score = np.array([[0]])

                # This is where the candidate lines are generated
                lst = search_back_meter(model, vocab, init_score,[w1],state, sess, 1,
                    self.words_to_pos, self.width, self.word_pools[word_pool_ind],
                    self.pos_to_words, template, template_sylls, self.dict_meters)
                # Sort each candidate line by score
                lst.sort(key=lambda x: x[0], reverse = True)
            else:
                raise IOError('No checkpoint')
        return template, lst

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
        third_line_sylls = first_line_sylls - 4
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
        *args : dict
            The parameters to be passed to the generation funciton.
        """
        for line, score, template in gen_func(*args):
            print('{:60} line score: {:2.3f}'.format(' '.join(line), score))
            print(template)
