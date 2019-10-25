from nltk.corpus import words
import pickle

word_list = words.words()
word_list = set(word_list)

whitelist = ["'s", 'has', 'jew', 'etc', 'tom', 'kid', 'jim', 'tv', 'uh', 'jr', 'dr', 'mrs', 'hid', 'tv',]
blacklist_index = set()
for index in range(50257):
    word = lg.enc.decode([index]).lower().strip()
    if word not in word_list and len(lg.get_word_pos(word)) > 0 \
    and 1 < len(word) < 4 and word not in whitelist:
        blacklist_index.add(index)
pickle.dump(blacklist_index, open("py_files/saved_objects/blacklist_index.p", "wb" ) )
