from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import word_tokenize, pos_tag
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords
import numpy as np
import re

class Meta_Poetry_Glove:
    glove_model = KeyedVectors.load_word2vec_format('~/Downloads/glove.6B/glove.6B.300d.w2v.txt', binary=False)
    glove_dim = len(glove_model['man'])
    punct = re.compile(r'[^\w\s]')
    ps = PorterStemmer()

    def __init__(self):
        self.already_seen = set()

    def get_glove_sim(self, w1, w2):
        """
        splits WordNet words or definitions
        and returns cosine similarity for
        the averages of all words in w1 and w2
        """
        split_re = re.compile(r'[\s|_]')
        avg_w1 = np.zeros(self.glove_dim)
        avg_w2 = np.zeros(self.glove_dim)
        for word in split_re.split(w1):
            if word not in self.glove_model:
                continue
            avg_w1 += self.glove_model.word_vec(word)
        avg_w1 /= np.sqrt(np.sum(avg_w1 ** 2))

        for word in split_re.split(w2):
            if word not in self.glove_model:
                continue
            avg_w2 += self.glove_model.word_vec(word)
        avg_w2 /= np.sqrt(np.sum(avg_w2 ** 2))
        return avg_w1.dot(avg_w2)

    def least_similar_glove(self, synsets):
        """
        finds two least similar synset
        among given synsets in Glove space
        """
        min_cos_sim = 1
        for i in range(len(synsets)):
            for j in range(i + 1, len(synsets)):
                word1 = remove_stopwords(self.punct.sub('', synsets[i].definition()))
                word2 = remove_stopwords(self.punct.sub('', synsets[j].definition()))
                this_sim = self.get_glove_sim(word1, word2)
                # assumes that Glove vocabulary has all words in wn,
                # may need to add error handling
                if this_sim < min_cos_sim:
                    min_cos_sim = this_sim
                    s1 = synsets[i]
                    s2 = synsets[j]
        return s1, s2

    def least_similar_glove_specify(self, synset):
        """
        returns synsets with lowest cosine
        similarity to specified synset in
        Glove space
        """
        word = synset.name().split('.')[0]
        min_sim = 1
        least_sim_synset = None
        for other_synset in wn.synsets(word):
            this_sim = self.get_glove_sim(synset.definition(), other_synset.definition())
            if this_sim < min_sim:
                min_sim = this_sim
                least_sim_synset = other_synset
        return self.get_sense_from_def(least_sim_synset), least_sim_synset.definition()

    def traverse_wn_glove(self, word):
        """
        finds most similar word among definitions
        of given words synsets
        """
        max_sim = -1
        best_word = None
        best_word_def = None
        for synset in wn.synsets(word):
            clean_def = remove_stopwords(self.punct.sub('', synset.definition()))
            for other_word in clean_def.split():
                this_sim = self.get_glove_sim(word, other_word)
                if self.ps.stem(other_word) not in self.already_seen and other_word != word and this_sim > max_sim:
                    max_sim = this_sim
                    best_word = other_word
                    best_word_def = synset.definition()
        return best_word, best_word_def

    def get_two_senses_glove(self, seed_word):
        """
        finds least similar synsets of seed word
        """
        synsets = wn.synsets(seed_word)
        pair = self.least_similar_glove(synsets)
        sense1 = self.get_sense_from_def(pair[0])
        sense2 = self.get_sense_from_def(pair[1])
        return (sense1, pair[0].definition()), (sense2, pair[1].definition())

    def get_sense_from_def(self, synset):
        """
        finds word that is most similar to given
        synset's name in Glove space
        """
        max_sim = -1
        best_word = None
        synset_word = synset.name().split('.')[0]
        # strip punctuation
        clean_def = remove_stopwords(self.punct.sub('', synset.definition()))
        for other_word in clean_def.split():
            if self.ps.stem(other_word) in self.already_seen:
                continue
            this_sim = self.get_glove_sim(synset_word, other_word)
            if other_word != synset_word and this_sim > max_sim:
                max_sim = this_sim
                best_word = other_word
        return best_word

    def five_word_algorithm_glove_specify(self, synset):
        """
        given a specific synset, traverse Wordnet to
        create five word outline for meta poetry
        """
        word_c = synset.name().split('.')[0], synset.definition()
        self.already_seen.add(self.ps.stem(word_c[0]))

        word_d = self.least_similar_glove_specify(synset)
        self.already_seen.add(self.ps.stem(word_d[0]))

        word_b = self.get_sense_from_def(synset)
        word_b = word_b, wn.synsets(word_b)[0].definition()
        self.already_seen.add(self.ps.stem(word_b[0]))

        word_a = self.traverse_wn_glove(word_b[0])
        self.already_seen.add(self.ps.stem(word_a[0]))

        word_e = self.traverse_wn_glove(word_d[0])

        self.already_seen.clear()

        return_list = []
        for word, definition in [word_a, word_b, word_c, word_d, word_e]:
            clean_def = set(definition.split())
            clean_def.discard(word)
            return_list.append((word, clean_def))

        return return_list

    def five_word_algorithm_glove(self, seed_word):
        """
        given a seed word, traverses Wordnet to create
        five word outline for meta poetry
        """
        word_c, word_d = self.get_two_senses_glove(seed_word)
        self.already_seen.add(self.ps.stem(word_c[0]))
        self.already_seen.add(self.ps.stem(word_d[0]))

        word_b = self.traverse_wn_glove(word_c[0])
        self.already_seen.add(self.ps.stem(word_b[0]))

        word_a = self.traverse_wn_glove(word_b[0])
        self.already_seen.add(self.ps.stem(word_a[0]))

        word_e = self.traverse_wn_glove(word_d[0])

        self.already_seen.clear()

        return_list = []
        for word, definition in [word_a, word_b, word_c, word_d, word_e]:
            clean_def = set(definition.split())
            clean_def.discard(word)
            return_list.append((word, clean_def))

        return return_list

    def print_five_words_glove(self, seed_word):
        words = self.five_word_algorithm_glove(seed_word)
        print(words[0][0] + '->' + words[1][0] + '->\033[4m' + words[2][0] +
              '\033[0m\033[1m~~>\033[0m\033[4m' + words[3][0] + '\033[0m->' + words[4][0])

    def print_five_words_glove_specify(self, synset):
        words = self.five_word_algorithm_glove_specify(synset)
        print(words[0][0] + '->' + words[1][0] + '->\033[4m' + words[2][0] +
              '\033[0m\033[1m~~>\033[0m\033[4m' + words[3][0] + '\033[0m->' + words[4][0])
