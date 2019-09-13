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
import copy
from functools import reduce
import math

from .model_back import Model as Model_back
from .functions import search_back_meter
from .templates import get_templates

from gpt2.src.score import score_model
from gpt2.src.generate_prompt import generate_prompt
from gpt2.src.encoder import get_encoder
from .templates import get_first_nnp, get_first_line_templates, get_good_templates


class Limerick_Generate:
    def __init__(self, wv_file='py_files/saved_objects/poetic_embeddings.300d.txt',
                 syllables_file='py_files/saved_objects/cmudict-0.7b.txt',
                 postag_file='py_files/saved_objects/postag_dict_all.p',
                 model_dir='py_files/models/all_combined_back',
                 model_name='345M', load_poetic_vectors=True):
        self.api_url = 'https://api.datamuse.com/words'
        self.ps = nltk.stem.PorterStemmer()
        self.punct = re.compile(r'[^\w\s]')
        self.model_dir = model_dir
        self.model_name = model_name
        self.poetic_vectors = KeyedVectors.load_word2vec_format(wv_file, binary=False) if load_poetic_vectors else None

        self.create_syll_dict(syllables_file)

        with open(postag_file, 'rb') as f:
            postag_dict = pickle.load(f)
        self.pos_to_words = postag_dict[1]
        self.words_to_pos = postag_dict[2]
        self.create_pos_syllables()
        self.create_templates_dict(postag_dict[0])

        self.first_line_words = pickle.load(open('py_files/saved_objects/first_line.p', 'rb'))
        self.width = 20
        # Not sure what this does, necessary for search_back function
        self.word_pools = [set([]) for n in range(4)]
        self.enc = get_encoder(self.model_name)
        # get male and female names
        # with open("py_files/saved_objects/dist.female.first.txt", "r") as hf:
        #     self.female_names = [lines.split()[0].lower() for lines in hf.readlines()]
        # with open("py_files/saved_objects/dist.male.first.txt", "r") as hf:
        #     self.male_names = [lines.split()[0].lower() for lines in hf.readlines()]
        # use filtered names instead
        with open("py_files/saved_objects/filtered_names.txt", "r") as hf:
            self.filtered_names = [line.split()[0].lower() for line in hf.readlines()]
        with open("py_files/saved_objects/filtered_nouns_verbs.txt", "r") as hf:
            self.filtered_nouns_verbs = [line.strip() for line in hf.readlines()]

        self.word_embedding_alpha = 0.5
        self.word_embedding_coefficient = 0.1

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
                                chars += "1"
                            else:
                                chars += ch
                newLine += [chars]
                lines[i] = newLine
                if(newLine[0] not in self.dict_meters):  # THIS IF STATEMENT ALLOWS FOR MULTIPLE PRONUNCIATIONS OF A WORD
                    self.dict_meters[newLine[0]] = [chars]
                else:
                    if(chars not in self.dict_meters[newLine[0]]):
                        self.dict_meters[newLine[0]] += [chars]
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

    def get_word_similarity(self, word, seed):
        """
        Given a seed word, if the word is a noun, verb or adjective, return the
        similarity between the two words. Otherwise, return None.

        Parameters
        ----------
        word : str
            A word in the poem. We want it to be similar in meaning to the last
            word of the line.
        seed : str
            The last word of the line.
        Returns
        -------
        float
            Word similarity between the word and the seed word.
        """
        if word not in self.words_to_pos or seed not in self.words_to_pos:
            return None
        word_pos = self.words_to_pos[word]
        if 'JJ' in word_pos \
            or 'NN' in word_pos \
            or any('VB' in pos for pos in word_pos):
            return self.poetic_vectors.similarity(word, seed)
        return None

    def is_duplicate_in_previous_words(self, word, previous):
        """
        Given a new word and previous words, if the word is a noun or adjective,
        and has appeared previously in the poem, return true. Otherwise, return
        false.
        Parameters
        ----------
        word : str
            New word we want to put into the poem.
        previous : list of str
            previous words in the poem.
        Returns
        -------
        bool
            Whether the word is a duplicate.
        """
        if word not in self.words_to_pos:
            return False
        word_pos = self.words_to_pos[word]
        if len(word_pos) == 0:
            return False
        return ('JJ' == word_pos[0] or 'NN' == word_pos[0]) and word in previous

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

    def get_similar_word(self, words, seen_words):
        """
        Given a list of words, return a word most similar to this list.
        """
        seen_words_set = set(seen_words) | set(words)

        word_set = set()

        for word in words:
            for synset in wn.synsets(word):
                clean_def = remove_stopwords(self.punct.sub('', synset.definition()))
                word_set.update(clean_def.lower().split())

        max_sim = -1
        best_word = None

        for other_word in word_set:
            if other_word not in self.poetic_vectors:
                continue

            sim = 0
            for word in words:
                sim += self.poetic_vectors.similarity(word, other_word) ** 0.5

            if sim > max_sim and self.ps.stem(other_word) not in seen_words_set:
                max_sim = sim
                best_word = other_word

        return best_word

    def get_similar_word_henry(self, words, seen_words=[], weights=1, n_return=1, word_set=None):
        """
        Given a list of words, return a list of words of a given number most similar to this list.
        <arg>:
        words: a list of words (prompts)
        seen_words: words not to repeat (automatically include words in arg <words> in the following code)
        weights: weights for arg <words>, default to be all equal
        n_return: number of words in the return most similar list
        word_set: a set of words to choose from, default set to the set of words extracted from the definitions of arg <word> in gensim
        <measure of similarity>:
        similarity from gensim squared and weighted sum by <arg> weights
        <return>:
        a list of words of length arg <n_return> most similar to <arg> words
        """
        seen_words_set = set(seen_words) | set(self.ps.stem(word) for word in words)

        if word_set is None:
            word_set = set()

            for word in words:
                for synset in wn.synsets(word):
                    clean_def = remove_stopwords(self.punct.sub('', synset.definition()))
                    word_set.update(clean_def.lower().split())
                word_set.update({dic["word"] for dic in requests.get(self.api_url, params={'rel_syn': "grace"}).json()})

        if weights == 1:
            weights = [1] * len(words)

        def cal_score(words, weights, syn):
            score = 0
            for word, weight in zip(words, weights):
                score += max(self.poetic_vectors.similarity(word, syn), 0) ** 0.5 * weight
            return score / sum(weights)

        syn_score_list = [(syn, cal_score(words, weights, syn)) for syn in word_set if self.ps.stem(syn) not in seen_words_set and syn in self.poetic_vectors]
        syn_score_list.sort(key=lambda x: x[1], reverse=True)

        return [e[0] for e in syn_score_list[:n_return]]

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
            if this_sim > max_sim and self.ps.stem(r['word']) not in seen_words:
                w3 = r['word']
                max_sim = this_sim

        if w5 is None or w3 is None or w1 is None:
            raise ValueError('Cannot generate limerick using ', w2)

        seen_words.add(self.ps.stem(w3))
        return w1, w2, w3, w4, w5

    def get_five_words_henry_old(self, w2, n_return=1):
        nouns = reduce(lambda x, y: x | y, [set(self.pos_to_words[tag]) for tag in ['NN', 'NNS', 'NNP', 'NNPS']])
        verbs = reduce(lambda x, y: x | y, [set(self.pos_to_words[tag]) for tag in ['VBG', 'VBZ', 'VBN', 'VBP', 'VB', 'VBD']])
        n_return_frac13 = math.ceil(n_return ** (1 / 3))

        def seen_words(words_list):
            return set(self.ps.stem(word) for word in words_list)

        def replicate(word_list, n_reps, max_reps):
            return [word for word in word_list for _ in range(n_reps)][:max_reps]

        # Three connection words
        w_response = requests.get(self.api_url, params={'rel_rhy': w2}).json()
#         rhyme_nn = set(d['word'] for d in w_response).intersection(nouns | verbs)
        rhyme_any = set(d['word'] for d in w_response)

        # w4
        w4s_raw = self.get_similar_word_henry([w2], n_return=n_return_frac13, word_set=(nouns | verbs))
        w4s = replicate(w4s_raw, n_return_frac13 ** 2, n_return)
        # print(w4s)

        # w3
        w3s_raw = []
        for w4 in w4s_raw:
            w_response2 = requests.get(self.api_url, params={'rel_rhy': w4}).json()
            rhyme_nn2 = set(d['word'] for d in w_response2).intersection(nouns | verbs)
            # print(w4, rhyme_nn2)
            w3s_raw.extend(self.get_similar_word_henry([w2, w4], weights=[3, 1], n_return=n_return_frac13, word_set=rhyme_nn2))
        w3s = replicate(w3s_raw, n_return_frac13 ** 2, n_return)
        # print(w3s)

        # w5
        w5s = []
        for w3, w4 in zip(w3s, w4s):
            w5s.append(self.get_similar_word_henry([w2, w3, w4], weights=[1, 2, 4], n_return=1, word_set=rhyme_any)[0])
        # print(w5s)

        # Find w1 that rhymes with w2 that is a city name / person name (to be changed soon)
        w1_list = []
        for r in rhyme_any:
            if r in self.words_to_pos:  # and self.ps.stem(r) not in seen_words():
                w1_list.append(r)

        w1s = [random.choice(w1_list) for _ in range(n_return)]

        if not (w1s and w3s and w4s and w5s):
            raise ValueError("Cannot generate limerick using " + w2)

        return list(zip(w1s, [w2] * n_return, w3s, w4s, w5s))

    def fill_three_words_henry(self, w1, rhyme_w1, w3, rhyme_w3, prompt):
        """
        given w1, w3 and prompt, as well as rhyming lists of w1 and w3, return the best choice of [w2, w4, w5] satisfying all constraints.
        <args>:
        w1: word 1, a name
        rhyme_w1: a list of words that rhyme with <arg> w1
        w3: word 3, a word that is usually similar in meaning to <arg> prompt
        rhyme_w3: a list of words that rhyme with <arg> w3
        prompt: the prompt word of the poem
        <method>:
        use self.get_similar_word_henry to select the best word from the rhyming lists, use random choice at word 2 to introduce randomness
        <return>:
        all five words [w1 (capitalized), w2, w3, w4, w5]
        """
        w2 = random.choice(self.get_similar_word_henry([prompt, w3], seen_words=[w1], weights=[2, 1], n_return=5, word_set=rhyme_w1))
        w4 = self.get_similar_word_henry([prompt, w2, w3], seen_words=[w1], weights=[1, 2, 4], n_return=1, word_set=rhyme_w3)[0]
        w5 = self.get_similar_word_henry([prompt, w2, w3, w4], seen_words=[w1], weights=[1, 4, 1, 3], n_return=1, word_set=rhyme_w1)[0]
        return [w1.capitalize(), w2, w3, w4, w5]

    def get_rhyming_words_one_step_henry(self, word, max_syllables=3):
        """
        get the rhyming words of <arg> word returned from datamuse api
        <args>:
        word: any word
        <return>:
        a set of words
        """
        return set(d['word'] for d in requests.get(self.api_url, params={'rel_rhy': word}).json() if " " not in d['word'] and d['word'] in self.poetic_vectors and
                   d['numSyllables'] <= max_syllables)

    def get_rhyming_words_henry(self, word, max_iter=10, max_words=1000, min_words=30):
        """
        get all rhyming words of <arg> word that could be found by datamuse api
        <args>:
        word: any word
        max_iter: maximum iteration number of broad-first-search
        max_words: maximum number of words in return
        <method>:
        do broad-first-search with self.get_rhyming_words_one_step_henry
        <return>:
        a set of words
        """
        rhyming_words = self.get_rhyming_words_one_step_henry(word)
        new_rhyming_words = rhyming_words.copy()
        count = 0
        print("'%s' bfs iteration 0: %d new words." % (word, len(new_rhyming_words)))
        while count < max_iter and len(rhyming_words) <= max_words:
            new_rhyming_words = {new_word for old_word in new_rhyming_words for new_word in self.get_rhyming_words_one_step_henry(old_word)
                                 if new_word not in (rhyming_words | new_rhyming_words | {word})}
            if len(new_rhyming_words) == 0:
                break
            print("'%s' bfs iteration %d: %d new words." % (word, count + 1, len(new_rhyming_words)))
            rhyming_words |= new_rhyming_words
            count += 1
        return rhyming_words

    def get_five_words_henry(self, prompt, rhyming_max_iter=4, rhyming_max_words=100):
        """
        given a prompt word, return a few dozens of storylines.
        <arg>:
        prompt: any word
        rhyming_max_iter: <arg> for self.get_rhyming_words_henry
        rhyming_max_words: <arg> for self.get_rhyming_words_henry
        <explanation>:
        <arg> rhyming_max_iter and <arg> rhyming_max_words balances the quality of storylines and the time it takes, usually (4, 100) is a good choice
        <method>:
        randomly choose 20 w1 from the male + female name lists that have rhyming words, and store these rhyming lists
        choose 20 w3 with meaning most similar to <arg> prompt, keep the ones that have rhyming words and store these rhyming lists
        generate storylines using self.fill_three_words_henry, if IndexError occurs (i.e. a list of length 0 is generated), then report the error
        <return>:
        a list of lists, each containing a storyline. The number of returned storylines is usually around 100, depending on <arg> prompt
        """
        names = [name for name in (self.male_names + self.female_names) if name[-1] not in "a"]
        w1s_rhyme_dict = {}
        w1_count = 0
        while w1_count < 20:
            w1 = random.choice(names)
            rhyme_w1 = self.get_rhyming_words_henry(w1, max_iter=rhyming_max_iter, max_words=rhyming_max_words)
            if len(rhyme_w1) == 0:
                print("Unable to find rhyming words of word 1 '%s'" % w1)
                continue
            w1s_rhyme_dict[w1] = rhyme_w1
            names.remove(w1)
            w1_count += 1
            print("Getting rhyming words of word 1 '%s' ...... %d / 20 done." % (w1, w1_count))

        nouns = reduce(lambda x, y: x | y, [set(self.pos_to_words[tag]) for tag in ['NN', 'NNS', 'NNP', 'NNPS']])
        verbs = reduce(lambda x, y: x | y, [set(self.pos_to_words[tag]) for tag in ['VBG', 'VBZ', 'VBN', 'VBP', 'VB', 'VBD']])

        w3s = self.get_similar_word_henry([prompt], n_return=10)
        w3s.extend(self.get_similar_word_henry([prompt], seen_words=w3s, n_return=10, word_set=(nouns | verbs)))
        w3s_rhyme_dict = {}
        for w3_count, w3 in enumerate(w3s):
            rhyme_w3 = self.get_rhyming_words_henry(w3, max_iter=rhyming_max_iter, max_words=rhyming_max_words)
            if len(rhyme_w3) == 0:
                print("Unable to find rhyming words of word 3 '%s'" % w3)
                continue
            w3s_rhyme_dict[w3] = rhyme_w3
            print("Getting rhyming words of word 3 '%s' ...... %d / 20 done." % (w3, w3_count + 1))

        storylines = []
        for w1, rhyme_w1 in w1s_rhyme_dict.items():
            for w3, rhyme_w3 in w3s_rhyme_dict.items():
                try:
                    storylines.append(self.fill_three_words_henry(w1, rhyme_w1, w3, rhyme_w3, prompt))
                except IndexError:
                    print("'%s' and '%s' are unable to generate storyline" % (w1, w3))
        return storylines

    def get_two_sets_henry(self, prompt, n_w125=20, n_w34=20):
        """
        <args>:
        prompt: prompt word
        n_w125: number of w1s in output
        n_w34: number of w3s in output
        <desc>:
        return a list of two dictionaries,
        dict1 has <arg> n_w125 keys as w1, and each value is a list of at least 30 words rhyming with w1, i.e. choices for w2, w5
        dict2 has <arg> n_w34 keys as w3, and each value is a list of at least 20 words rhyming with w3, i.e. choices for w4
        <return>:
        [ w1s_rhyme_dict: {w1: [w2/w5s]}, [w2/w5s] containing at least 30 words ,
        w3s_rhyme_dict: {w3: [w4s]}, [w4s] containing at least 20 words ]
        """
        w1s = random.sample(self.filtered_names, n_w125)
        w1s_rhyme_dict = {w1: set(self.get_rhyming_words_one_step_henry(w1)) for w1 in w1s}

        w3s = self.get_similar_word_henry([prompt], seen_words=w1s, n_return=n_w34, word_set=set(self.filtered_nouns_verbs))
        w3s_rhyme_dict = {w3: set(self.get_rhyming_words_one_step_henry(w3)) for w3 in w3s}

        return w1s_rhyme_dict, w3s_rhyme_dict

    def get_all_partition_size_n(self, num_sylls, template, last_word_sylls):
        """
        Returns all integer partitions of an int with a partition_size number
        of ints.
        """
        def get_all_partitions(n, I=1):
            yield (n,)
            for i in range(I, n // 2 + 1):
                for p in get_all_partitions(n - i, i):
                    yield (i,) + p

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
        all_partitions = [list(itertools.permutations(p)) for p in get_all_partitions(syllables_left) if len(p) == num_words_left]
        ret = set()
        for permutation in all_partitions:
            for partition in permutation:
                if valid_syll(partition, template):
                    ret.add(partition)
        ret = list(ret)
        random.shuffle(ret)
        return ret

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
                for i in range(I, n // 2 + 1):
                    for p in get_all_partitions(n - i, i):
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

    def load_name_list(self, name_count=None):
        female_names_file = 'py_files/saved_objects/dist.female.first.txt'
        male_names_file = 'py_files/saved_objects/dist.male.first.txt'
        female_name_list, male_name_list = [], []

        with open(female_names_file, 'rb') as female_names:
            for line in female_names:
                name, _, _, count = line.split()
                female_name_list.append(name.lower().decode('utf-8'))
                if name_count is not None and int(count) == name_count:
                    break

        with open(male_names_file, 'rb') as male_names:
            for line in male_names:
                name, _, _, count = line.split()
                male_name_list.append(name.lower().decode('utf-8'))
                if name_count is not None and int(count) == name_count:
                    break

        return female_name_list, male_name_list

    def load_city_list(self):
        city_list_file = 'py_files/saved_objects/city_names.txt'
        l = []
        with open(city_list_file, 'rb') as cities:
            for line in cities:
                l.append(line.rstrip().decode('utf-8').lower())
        return l

    # For instance, if correct meter is: da DUM da da DUM da da DUM, pass in
    # stress = [1,4,7] to enforce that the 2nd, 5th & 8th syllables have stress.
    def is_correct_meter(self, template, num_syllables=[8], stress=[1, 4, 7]):
        meter = []
        n = 0
        for x in template:
            if x not in self.dict_meters:
                return False
            n += len(self.dict_meters[x][0])
            curr_meter = self.dict_meters[x]
            for i in range(len(curr_meter[0])):
                curr_stress = []
                for possible_stress in curr_meter:
                    curr_stress.append(possible_stress[i])
                meter.append(curr_stress)
        return (not all(('1' not in meter[i]) for i in stress)) \
            and (n in num_syllables)

    def gen_first_line_new(self, last_word, contains_adjective=True, strict=False, search_space=100):
        """
        Generetes all possible first lines of a Limerick by going through a
        set of template. Number of syllables is always 8 or 9.

        Parameters
        ----------
        w1 : str
            The last word in the line, used to generate backwards from.
        last_word : str
            The last word of the first_line sentence specified by the user.
        strict : boolean, optional
            Set to false by default. If strict is set to false, this method
            will look for not only sentences that end with last_word, but also
            sentences that end with a word that rhyme with last_word.

        Returns
        -------
        first_lines : list
            All possible first line sentences.
        """

        def get_num_sylls(template):
            n = 0
            for x in template:
                if x not in self.dict_meters:
                    return 0
                n += len(self.dict_meters[x][0])
            return n

        female_name_list, male_name_list = self.load_name_list()
        city_name_list = self.load_city_list()
        templates, placeholders, dict = get_first_line_templates()

        if strict:
            if last_word not in female_name_list and \
                last_word not in male_name_list and \
                last_word not in city_name_list:
                raise Exception('last word ' + last_word + ' is not a known name or location')
            last_word_is_location = last_word in city_name_list
            last_word_is_male = last_word in male_name_list
            last_word_is_female = last_word in female_name_list

        w_response = requests.get(self.api_url, params={'rel_rhy': last_word}).json()
        w_response = set(d['word'] for d in w_response)
        w_response.add(last_word)
        candidate_sentences = []

        for template in templates:
            if strict and last_word_is_location and template[-1] != 'PLACE':
                continue
            if strict and (last_word_is_male or last_word_is_female) and \
                template[-1] != 'NAME':
                continue
            if not contains_adjective and ('JJ' in template):
                continue
            candidates = []
            for word in template:
                if word not in placeholders:
                    continue
                if word == 'PERSON':
                    person_dict = dict['PERSON']
                    if len(candidates) == 0:
                        candidates = [{'PERSON': p, 'GENDER': 'MALE'} for p in person_dict['MALE']] \
                            + [{'PERSON': p, 'GENDER': 'FEMALE'} for p in person_dict['FEMALE']] \
                            + [{'PERSON': p, 'GENDER': 'NEUTRAL'} for p in person_dict['NEUTRAL']]
                    else:
                        new_candidates = []
                        for d in candidates:
                            for gender in person_dict:
                                for p in person_dict[gender]:
                                    new_d = copy.deepcopy(d)
                                    new_d['PERSON'] = p
                                    new_d['GENDER'] = gender
                                    new_candidates.append(new_d)
                        candidates = new_candidates
                if word == 'JJ':
                    adj_dict = dict['JJ']
                    if len(candidates) == 0:
                        candidates = [{'JJ': w} for w in adj_dict]
                    else:
                        new_candidates = []
                        for d in candidates:
                            for adj in adj_dict:
                                new_d = copy.deepcopy(d)
                                new_d['JJ'] = adj
                                new_candidates.append(new_d)
                        candidates = new_candidates
                if word == 'IN':
                    in_dict = dict['IN']
                    new_candidates = []
                    for d in candidates:
                        for i in in_dict:
                            new_d = copy.deepcopy(d)
                            new_d['IN'] = i
                            new_candidates.append(new_d)
                    candidates = new_candidates
                if word == 'PLACE':
                    if strict and last_word_is_location:
                        for d in candidates:
                            d['PLACE'] = last_word
                    new_candidates = []
                    for d in candidates:
                        for city in city_name_list:
                            new_d = copy.deepcopy(d)
                            new_d['PLACE'] = city
                            new_candidates.append(new_d)
                    candidates = new_candidates
                if word == 'NAME':
                    # Only select candidates with the correct gender as the name
                    if strict and (last_word_is_male or last_word_is_female):
                        new_candidates = []
                        for d in candidates:
                            if d['GENDER'] == 'FEMALE' and last_word_is_female:
                                d['NAME'] = last_word
                                new_candidates.append(d)
                            elif d['GENDER'] == 'MALE' and last_word_is_male:
                                d['NAME'] = last_word
                                new_candidates.append(d)
                            elif d['GENDER'] == 'NEUTRAL':
                                d['NAME'] = last_word
                                new_candidates.append(d)
                        candidates = new_candidates
                        continue

                    new_candidates = []
                    for d in candidates:
                        if d['GENDER'] == 'MALE' or d['GENDER'] == 'NEUTRAL':
                            for name in male_name_list:
                                new_d = copy.deepcopy(d)
                                new_d['NAME'] = name
                                new_candidates.append(new_d)
                        if d['GENDER'] == 'FEMALE' or d['GENDER'] == 'NEUTRAL':
                            for name in female_name_list:
                                new_d = copy.deepcopy(d)
                                new_d['NAME'] = name
                                new_candidates.append(new_d)
                    candidates = new_candidates
            for candidate in candidates:
                if candidate[template[-1]] not in w_response:
                    continue
                new_sentence = copy.deepcopy(template)
                for i in range(len(new_sentence)):
                    if new_sentence[i] in placeholders:
                        new_sentence[i] = candidate[new_sentence[i]]
                # First line always has 8 or 9 syllables
                if self.is_correct_meter(new_sentence, num_syllables=[8, 9]):
                    candidate_sentences.append(new_sentence)
        random.shuffle(candidate_sentences)
        return candidate_sentences[:search_space]

    def gen_first_line(self, w2, num_sylls):
        def get_num_sylls(template):
            n = 0
            for x in template:
                n += len(self.dict_meters[x][0])
            return n

        names = self.first_line_words[0]
        cities = self.first_line_words[1]
        names = {x[0]: x[1] for x in names}
        w_response = requests.get(self.api_url, params={'rel_rhy': w2}).json()
        rhyme_names = set(d['word'] for d in w_response).intersection(names.keys())
        rhyme_cities = set(d['word'] for d in w_response).intersection(cities)
        templates = get_first_nnp()
        possible_sentence = []
        for name in rhyme_names:
            for template in templates[names[name]]:
                print(name)
                if len(self.dict_meters[name][0]) + get_num_sylls(template) == num_sylls:
                    possible_sentence.append(template + [name])
        for name in rhyme_cities:
            for template in templates['city']:
                try:
                    if len(self.dict_meters[name][0]) + get_num_sylls(template) == num_sylls:
                        possible_sentence.append(template + [name])
                except:
                    continue
        if len(possible_sentence) == 0:
            raise ValueError('No lines can be constructed with this metric')
        else:
            return possible_sentence

    def gen_line(self, w1, template=None, num_sylls=10, state=None, score=None):
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
            pos = self.words_to_pos[w1][0]
        if templates is None and set_of_templates is not None:
            try:
                t = set_of_templates[pos]
            except KeyError:
                print('No templates for POS')
                raise ValueError('No lines can be constructed')
            n_templates = min(len(t), rand_templates)
            templates = random.sample(t, k=n_templates)
            # template = self.get_rand_template(num_sylls, w1)
            # temp_len = random.randint(num_sylls - last_word_sylls - 1, num_sylls - last_word_sylls)
            # template = random.choice(self.templates_dict[(w1_pos, temp_len)])

        # Assign syllables to each pos in template
        lines = []
        for template in templates:
            try:
                t, line = self.gen_line(w1, template=template[0], num_sylls=num_sylls, state=state, score=score)
                this_line = line[0][1][1]
                this_score = line[0][0].item() / len(this_line)
                if return_state:
                    lines.append((this_line, this_score, t, template[1], line[0][1][0][1]))
                else:
                    lines.append((this_line, this_score, t, template[1]))
            except:
                continue
        lines.sort(key=lambda x: x[1], reverse=True)
        if len(lines) == 0:
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

    def gen_poem_independent_matias(self, seed_word, first_line_sylls, rand_template=5, storyline=0):
        def get_templates_last(n, key):
            _, _1, _2, data = get_templates()
            df = data[key]
            min_n = min(n, len(df))
            t = random.sample(df, k=min_n)
            fourth = []
            fifth = []
            for template in t:
                fourth.append((template[0][:template[2] + 1], template[1][:template[2] + 1]))
                fifth.append((template[0][template[2] + 1:], template[1][template[2] + 1:]))
            return fourth, fifth

        five_words = self.get_five_words_henry(seed_word) if storyline else self.get_five_words(seed_word)
        first_line = random.choice(self.gen_first_line(seed_word, first_line_sylls))

        lines = [[first_line]]
        third_line_sylls = first_line_sylls - 4

        dataset, second_line_, third_line_, last_two_lines = get_templates()
        # templates = []
        # try:
        # templates 2nd line:
        # templates.append(random.choice(second_line_[self.words_to_pos[five_words[1]][0]]))
        # templates 3rd line
        # templates.append(random.choice(third_line_[self.words_to_pos[five_words[2]][0]]))
        # templates 4th line
        key = self.words_to_pos[five_words[3]][0] + '-' + self.words_to_pos[five_words[4]][0]
        # temp=random.choice(last_two_lines[key])
        # templates.append((temp[0][:temp[2]+1], temp[1][:temp[2]+1]))
        # templates 5th line
        # templates.append((temp[0][temp[2]+1:], temp[1][temp[2]+1:]))
        # except:
        #    print('POS Not in dataset of templates')
        #    return None
        fourth, fifth = get_templates_last(rand_template, key)
        for i, w in enumerate(five_words):
            # Set number of syllables from generated line dependent on which
            # line is being generated
            if i == 0:
                continue
            elif i == 1:
                this_line_sylls = first_line_sylls
                out = self.gen_best_line(w, num_sylls=this_line_sylls, set_of_templates=second_line_)
            elif i == 2:
                this_line_sylls = third_line_sylls
                out = self.gen_best_line(w, num_sylls=this_line_sylls, set_of_templates=third_line_)
            elif i == 3:
                out = self.gen_best_line(w, num_sylls=third_line_sylls, templates=fourth)
            elif i == 4:
                out = self.gen_best_line(w, num_sylls=first_line_sylls, templates=fifth)
            if out is None or out == []:
                raise ValueError
            print(out)
            lines.append(out[0])
        print("************")
        string = ''
        for x in lines:
            string += ' '.join(x[0]) + '\n'
        print(string)
        return lines

    def gen_poem_conditioned(self, seed_word, second_line_sylls, rand_template=5):
        five_words = self.get_five_words(seed_word)
        print('five words are: ')
        print(five_words)

        def get_templates_last(n, key):
            _, _1, _2, data = get_templates()
            df = data[key]
            min_n = min(n, len(df))
            t = random.sample(df, k=min_n)
            fourth = []
            fifth = []
            for template in t:
                fourth.append((template[0][:template[2] + 1], template[1][:template[2] + 1]))
                fifth.append((template[0][template[2] + 1:], template[1][template[2] + 1:]))
            return fourth, fifth

        dataset, second_line_, third_line_, last_two_lines = get_templates()
        # t_2=random.choice(second_line_[self.words_to_pos[five_words[1]][0]])[0]
        key = self.words_to_pos[five_words[3]][0] + '-' + self.words_to_pos[five_words[4]][0]
        fourth, fifth = get_templates_last(rand_template, key)

        # t=random.choice(last_two_lines[key])
        # t_4=t[0][:t[2]+1]
        # t_5=t[0][t[2]+1:]
        # t_1 = random.choice(dataset[self.words_to_pos[five_words[0]][0]])[0]
        # t_3 = random.choice(third_line_[self.words_to_pos[five_words[2]][0]])[0]

        o2 = self.gen_best_line(five_words[1], num_sylls=second_line_sylls, set_of_templates=second_line_)
        line2 = o2[0][0]
        score2 = o2[0][1]
        # state2 = o2[0][1][0][1]
        o3 = self.gen_best_line(five_words[2], num_sylls=second_line_sylls - 3, set_of_templates=third_line_)
        line3 = o3[0][0]
        score3 = o3[0][0]
        last = []
        for line_4, line_5 in zip(fourth, fifth):
            o5 = self.gen_best_line(five_words[4], num_sylls=second_line_sylls, templates=[line_5], return_state=True)
            line5 = o5[0][0]
            score5 = o5[0][1]  # / len(line5)
            state5 = o5[0][-1]
            score_for4, state_for4 = self.compute_next_state(state5, score5, line5)
            o4 = self.gen_best_line(five_words[3], num_sylls=second_line_sylls - 3, templates=[line_4], state=state_for4, score=score_for4)
            try:
                last.append((o4[0], (line5, score5, line_5[2]), o4[1]))
            except:
                continue
        last.sort(key=lambda x: x[2], reverse=True)
        if len(last) == 0:
            raise ValueError('no lines can be constructed')
        line4 = last[0][0][0]
        score4 = last[0][0][1]
        line5 = last[0][1][0]
        score5 = last[0][1][1]
        # score_for4, state_for4=self.compute_next_state(state5, score5, line5)
        # o1 = self.run_gen_model_back(line2, t1, second_line_sylls, state=state2, score=score2)
        # t1, o1=self.gen_line(five_words[0], t_1,num_sylls=second_line_sylls, state=state_for1, score=score_for1)

        # line1 = o1[0][1][1]
        # score1 = o1[0][0].item() / len(line1)

        # t4, o4=self.gen_line(five_words[3], t_4,num_sylls=second_line_sylls-3, state=state_for4, score=score_for4)
        # o3 = self.run_gen_line(line4, t3, second_line_sylls - 3, state=state4, score=score4)
        # line4 = o4[0][1][1]
        # score4 = o4[0][0].item() / len(line4)

        lines = []
        for i in range(2, 6):
            num_as_str = str(i)
            this_line = (locals()['line' + num_as_str], locals()['score' + num_as_str], locals()['t' + num_as_str])
            lines.append(this_line)
        print("************")
        string = ''
        for x in lines:
            string += ' '.join(x[0]) + '\n'
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
            key = np.random.choice(list(dataset.keys()), 1, p=[len(dataset[key]) / s for key in dataset.keys()])
            template = dataset[key[0]][random.randint(0, len(dataset[key[0]]))][0]

        new_line = []
        new_line_tokens = []
        for e in w.lower().split():
            new_line_tokens.append(self.enc.encode(e)[0])
        w_response = requests.get(self.api_url, params={'rel_rhy': rhyme}).json()
        rhyme_set = set(d['word'] for d in w_response)
        for i in range(len(template)):
            # Logits is the output of GPT model, encoder is used to decode the output
            logits = score_model(model_name=self.model_name, context_token=[new_line_tokens])
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
                     rhyme_word=None, rhyme_set=None, search_space=100, num_sylls=[], stress=[],
                     use_nltk=False, use_word_embedding=False):
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
        search_space : int, optional
            Search space for the GPT2 model
        num_sylls : array of ints, optional
            Number of syllables in the word
        stress : array of ints, optional
            Positions of stress of the sentence, corresponding to '1' in cmudict translation
        use_nltk: bool, optional
            If set to true, for words that don't appear in POS dictionary, the script will generate
            the POS in NLTK. It will be slower, but it allows for a larger set of words
        use_word_embedding: bool, optional
            If set to true, the method will calculate the word embedding distance between the current
            word and the last word in the storyline and take this into consideration when calculating
            the fitness score of a sentence. If you are not using storyline do not set this option
            to true.
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

        if not rhyme_set and rhyme_word:
            w_response = requests.get(self.api_url, params={'rel_rhy': rhyme_word}).json()
            rhyme_set = set(d['word'] for d in w_response)
            # Include the word itself in the rhyme set
            rhyme_set.add(rhyme_word)

        # Tuple format: (original word array, encode array, log probability of this sentence,
        # number of syllables, word embedding moving average (optional))
        if w:
            sentences = [(w.lower().split(), [], 0, 0)]
            for e in w.lower().split():
                sentences[0][1].append(self.enc.encode(e)[0])
        if encodes:
            sentences = [([], encodes, 0, 0, 0)]

        for i in range(len(template)):
            # Logits is the output of GPT model, encoder is used to decode the output
            logits = score_model(model_name=self.model_name, context_token=[s[1] for s in sentences])
            POS = template[i]

            new_sentences = []
            # For each sentence, calculate probability of a new word
            for j in range(len(sentences)):
                # There might be duplicate words such as "And" and " and" and we only need one
                for index in range(len(logits[j])):
                    word = self.enc.decode([index]).lower().strip()
                    if use_nltk:
                        if len(word.lower().strip()) == 0:
                            fit_pos = False
                        elif len(self.words_to_pos[word.lower().strip()]) == 0:
                            word_pos = nltk.pos_tag([word.lower().strip()])
                            if len(word_pos) == 0:
                                fit_pos = False
                            else:
                                fit_pos = POS == word_pos[0][1]
                        else:
                            fit_pos = POS in self.words_to_pos[word]
                    else:
                        fit_pos = POS in self.words_to_pos[word]
                    # Restrict the word to have the POS of the template
                    if fit_pos:
                        stripped_word = word.lower().strip()

                        # Enforce rhyme if last word
                        if i == len(template) - 1 and rhyme_set and (stripped_word not in rhyme_set):
                            continue
                        # Enforce meter if meter is provided
                        syllables = sentences[j][3]
                        if len(num_sylls) > 0 or len(stress) > 0:
                            if stripped_word not in self.dict_meters:
                                continue

                            possible_syllables = self.dict_meters[stripped_word]
                            word_length = min(len(s) for s in possible_syllables)

                            # Check if the entire line meets the syllables constraint
                            if i == len(template) - 1 and syllables + word_length not in num_sylls:
                                continue

                            # Check if the new word meets the stress constraint
                            if len(stress) > 0:
                                correct_stress = True
                                # There is a stress on current word
                                for stress_position in stress:
                                    if syllables <= stress_position and syllables + word_length > stress_position:
                                        stress_syllable_pos = stress_position - syllables
                                        if all(s[stress_syllable_pos] != '1' for s in possible_syllables):
                                            correct_stress = False
                                        break
                                if not correct_stress:
                                    continue
                            # Add current word's number of syllables to
                            # the sentence's number of syllables
                            syllables += word_length
                        moving_average = 0
                        if use_word_embedding:
                            # Calculate word embedding moving average with the story line set selection
                            embedding_distance = max(self.get_word_similarity(word, rhyme) for rhyme in rhyme_set)
                            moving_average = (1 - self.word_embedding_alpha) * sentences[j][4] + self.word_embedding_alpha * embedding_distance \
                                             if embedding_distance is not None \
                                             else sentences[j][4]
                        # Add candidate sentence to new array
                        new_sentences.append((sentences[j][0] + [word],
                                              sentences[j][1] + [index],
                                              sentences[j][2] + np.log(logits[j][index]),
                                              syllables,
                                              moving_average))

            # Get the most probable N sentences by sorting the list according to probability
            sentences = heapq.nsmallest(min(len(new_sentences), search_space),
                                        new_sentences,
                                        key = lambda x: -x[2] - self.word_embedding_coefficient * moving_average)
        print(sentences[0][0])
        return sentences[0]

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

        if not rhyme_set and rhyme_word:
            w_response = requests.get(self.api_url, params={'rel_rhy': rhyme_word}).json()
            rhyme_set = set(d['word'] for d in w_response)
            # Include the word itself in the rhyme set
            rhyme_set.add(rhyme)  # ?????????? undefined <var> rhyme? ??????????

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

    def gen_poem_gpt(self, rhyme1, rhyme2, default_templates=None,
                     story_line=False, prompt_length=100, save_as_pickle=False, search_space=100,
                     enforce_syllables=False, enforce_stress=False, search_space_coef=[1, 1, 0.5, 0.25],
                     use_word_embedding=False):
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
        search_space_coef: float, optional
            Decay rate of search space.The more sentences we run, the longer the prompt is.
            Setting the decay rate to be less than 1 limits the search space of the last
            couple sentences.
        use_word_embedding: float, optional

        Returns
        -------
        A string array containing the generated poem
        """

        if not default_templates:
            default_templates = random.choice(get_good_templates())

        if story_line:
            two_sets = self.get_two_sets_henry(rhyme1)
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
        out = generate_prompt(model_name=self.model_name, seed_word=rhyme1, length=prompt_length)
        prompt = self.enc.decode(out[0][0])
        prompt = prompt[:prompt.rfind(".") + 1]

        if story_line:
            rhyme1 = random.choice(list(two_sets[0].keys()))
        first_line = random.choice(self.gen_first_line_new(rhyme1))
        print(first_line)
        first_line_encodes = self.enc.encode(" ".join(first_line))
        prompt = self.enc.encode(prompt) + first_line_encodes

        if not story_line:
            r1_set.discard(first_line[-1])

        # Option to save the prompt in a file and generate sentences in different runs
        if save_as_pickle:
            with open('gpt2.pkl', 'wb') as f:
                pickle.dump(prompt, f)
            return

        generated_poem = [first_line]

        w0 = rhyme1
        w2 = None
        for i in range(4):
            if enforce_syllables:
                curr_sylls = [5, 6] if (i == 2 or i == 3) else [8, 9]
            else:
                curr_sylls = []

            if enforce_stress:
                stress = [1, 4] if (i == 2 or i == 3) else [1, 4, 7]
            else:
                stress = []

            if not story_line:
                rhyme_set = r1_set if (i == 0 or i == 3) else r2_set
                new_sentence = self.gen_line_gpt(w=None, encodes=prompt,
                                                 default_template=default_templates[i], rhyme_set=rhyme_set,
                                                 search_space=int(search_space * search_space_coef[i]),
                                                 num_sylls=curr_sylls, stress=stress, use_word_embedding=use_word_embedding)

                rhyme_set.discard(new_sentence[0][-1])
            else:
                if i == 0:
                    rhyme_set = two_sets[0][w0]
                elif i == 1:
                    rhyme_set = two_sets[1].keys()
                elif i == 2:
                    rhyme_set = two_sets[1][w2]
                elif i == 3:
                    rhyme_set = two_sets[1][w0]
                new_sentence = self.gen_line_gpt(w=None, encodes=prompt,
                                                 default_template=default_templates[i], rhyme_set=rhyme_set,
                                                 search_space=int(search_space * search_space_coef[i]),
                                                 num_sylls=curr_sylls, stress=stress, use_word_embedding=use_word_embedding)
                last_word = new_sentence[0][-1]
                if i == 0:
                    two_sets[0][w0].discard(last_word)
                elif i == 1:
                    s2 = last_word
            prompt += new_sentence[1]
            generated_poem.append(new_sentence[0])
        return generated_poem


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

        pos_length = {}
        for i in self.pos_to_words.keys():
            pos_length[i] = len(self.pos_to_words[i])

        words = re.sub("[^\w]", " ", prompt).split()
        for word in words:
            for POS in self.words_to_pos[word.lower()]:
                word_dict[POS].add(word.lower())
        for POS in word_dict.keys():
            word_dict[POS] = list(word_dict[POS])

        results = []
        encodes = []

        sentences = [['he', 'would', 'go', 'to', 'a', 'party'], ['i', 'can', 'stare', 'at', 'the', 'sky']]
        for i in sentences:
            temp = [self.enc.encode(word)[0] for word in i]
            encodes.append(temp)

        for i in range(num):
            sentence = []
            for POS in template:
                if pos_length[POS] <= 50:
                    w = random.choice(self.pos_to_words[POS])
                else:
                    w = random.choice(word_dict[POS])
                sentence.append(w)
            sentences.append(sentence)
            encodes.append([self.enc.encode(word)[0] for word in sentence])
        encodes = np.array(encodes)

        probs = np.zeros(len(sentences))

        for j in tqdm.trange(1, len(sentences[0])):
            results = score_model(model_name=self.model_name, context_token=encodes[:, :j])
            for i in range(len(sentences)):
                probs[i] += np.log(results[i][encodes[i][j]])

        index = np.argsort(np.negative(probs))
        for i in index:
            print("{}: {}".format(probs[i], sentences[i]))

        # print(sentences)
        # print(probs)
        return
