from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag


def least_similar(synsets):
    # returns entries with highest wup similarity score
    the_max = 0
    for i in range(len(synsets)):
        for j in range(i + 1, len(synsets)):
            this_similarity = synsets[i].wup_similarity(synsets[j])
            if this_similarity is not None and this_similarity > the_max:
                the_max = this_similarity
                s1 = synsets[i]
                s2 = synsets[j]
    return (s1, s2)

def best_corresponding_pos(synset):
    # inspects definition and returns most similar word with
    # same pos as input word excluding the input word
    # right now only works for nouns, verbs, adjectives
    this_pos = synset.pos()
    def_token = word_tokenize(synset.definition())
    
    if  this_pos == wn.NOUN:
        pos_token = 'NN'
    elif this_pos == wn.VERB:
        pos_token = 'V'
    elif this_pos == wn.ADJ:
        pos_token = 'JJ'
    else:
        raise ValueError('Input synset must be a Noun, Verb, or Adjective')
        
    lm = WordNetLemmatizer()
    max_similarity = -1
    best_corr_pos = None
    
    for tagged_word in pos_tag(def_token):
        if pos_token in tagged_word[1]:
            lemma = lm.lemmatize(tagged_word[0], pos=synset.pos())
            other_synsets = wn.synsets(lemma, pos=synset.pos())
            if len(other_synsets) > 0:
                # just pick the first synset
                this_similarity = synset.wup_similarity(other_synsets[0])
                if this_similarity > max_similarity:
                    max_similarity = this_similarity
                    best_corr_pos = tagged_word[0]
    return best_corr_pos

def first_corresponding_pos(synset):
    # inspects definition and returns first word with same pos
    # as input word
    # right now only works for nouns, verbs, adjectives
    this_pos = synset.pos()
    def_token = word_tokenize(synset.definition())
    
    if this_pos == 'n':
        pos_token = 'NN'
    elif this_pos == 'v':
        pos_token = 'V'
    elif this_pos == 'a':
        pos_token = 'JJ'
    else:
        raise ValueError('Input synset must be a Noun, Verb, or Adjective')
    
    for tagged_word in pos_tag(def_token):
        if pos_token in tagged_word[1]:
            return tagged_word[0]
        
def get_two_senses(seed_word):
    synsets = wn.synsets(seed_word)
    pair = least_similar(synsets)
    return best_corresponding_pos(pair[0]), best_corresponding_pos(pair[1])
#     return first_corresponding_pos(pair[0]), first_corresponding_pos(pair[1])

def traverse_wn(word):
    # traverses wn synsets for word and returns first
    # word in definition of synsets with same pos
    for synset in wn.synsets(word):
        first_pos = best_corresponding_pos(synset)
#         first_pos = first_corresponding_pos(synset)
        if first_pos is not None and first_pos != word:
            return first_pos
        
def five_word_algorithm(seed_word):
    word_c, word_d = get_two_senses(seed_word)
    word_b = traverse_wn(word_c)
    word_a = traverse_wn(word_b)
    word_e = traverse_wn(word_d)
    return word_a, word_b, word_c, word_d, word_e

def print_five_words(seed_word):
    words = five_word_algorithm(seed_word)
    print(words[0] + '->' + words[1] + '->\033[4m' + words[2] + 
          '\033[0m\033[1m~~>\033[0m\033[4m' + words[3] + '\033[0m->' + words[4])
    return words


def bool_five_words(list_words, vocab):
    for i, word in enumerate(list_words):
        if word not in set(list(vocab.keys())) and i!=3:
            return False
    return True
def create_pairs(lw):
    return [[lw[0], lw[1]],[lw[1],lw[2]],[lw[3], lw[4]],[lw[4], lw[0]]]