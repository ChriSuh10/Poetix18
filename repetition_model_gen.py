#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
# Open a file: file
sent_table = pd.read_csv('data/corpus.txt', delimiter='|', usecols=['text'])
#print(sent_table)


for i in sent_table.index.values:
    #print(sent_table.iloc[i]['text']);
    #sent_table.iloc[i]['text'] = sent_table.iloc[i]['text'].replace('''O ''', '''Oh''')
    sent_table.iloc[i]['text'] = sent_table.iloc[i]['text'].replace('''`''', ''' \'''')
    sent_table.iloc[i]['text'] = sent_table.iloc[i]['text'].replace(''' ` ''', '''\'''')
    sent_table.iloc[i]['text'] = sent_table.iloc[i]['text'].replace(''' `''', '''\'''')
    sent_table.iloc[i]['text'] = sent_table.iloc[i]['text'].replace(''' \'''','''\'''')
    sent_table.iloc[i]['text'] = sent_table.iloc[i]['text'].replace('''\'d''','''ed''')
    #sent_table.iloc[i]['text'] = re.sub(r'''[^\w\d'\-\s]+''','',sent_table.iloc[i]['text']);
    sent_table.iloc[i]['text'] = sent_table.iloc[i]['text'].strip();


#%%
def uniq(input):
  output = []
  for x in input:
    if x not in output:
      output.append(x)
  return output

def recursive_len(item):
    if type(item) == list:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 1


# In[2]:


import pprint
import nltk
import numpy as np
import time
from datetime import datetime
from enum import Enum
import json
num_figs=10
max_fig_length=18
base_path = 'data/rhet/repitition/'
base_path_output = 'data/rhet/output/'
corpus = 'sonnets'

fig_names = ["Anadiplosis", "Anaphora", "Antimetabole", "Conduplicatio", "Epanalepsis", "Epistrophe", "Epizeuxis", "Ploce", "Polysyndeton", "Symploce"]

class RHETORICAL_FIGURES(Enum):
    ANADIPLOSIS = 0
    ANAPHORA = 1
    ANTIMETABOLE = 2
    CONDUPLICATIO = 3
    EPANALEPSIS = 4
    EPISTROPHE = 5
    EPIZEUXIS = 6
    PLOCE = 7
    POLYSYNDETON = 8
    SYMPLOCE = 9

#FIGURE_DESCRIPTION[RHETORICAL_FIGURES['ANADIPLOSIS'].name].value
class FIGURE_DESCRIPTION(Enum):
    ANADIPLOSIS = 'Repetition of the ending word or phrase from the previous clause at the beginning of the next.'
    ANAPHORA = 'Repetition of a word or phrase at the beginning of successive phrases or clauses.'
    ANTIMETABOLE = 'Repetition of words in reverse grammatical order.'
    CONDUPLICATIO = 'The repetition of a word or phrase.'
    EPANALEPSIS = 'Repetition at the end of a clause of the word or phrase that began it.'
    EPISTROPHE = 'Repetition of the same word or phrase at the end of successive clauses.'
    EPIZEUXIS = 'Repetition of a word or phrase with no others between.'
    PLOCE = 'The repetition of word in a short span of text for rhetorical emphasis.'
    POLYSYNDETON = '"Excessive" repetition of conjunctions between clauses.'
    SYMPLOCE = 'Repetition of a word or phrase at the beginning, and of another at the end, of successive clauses.'

class FigureInfo:

    def __init__(self, unique_id, orig_text, pos_template, repitition_template):
        self.unique_id = unique_id
        self.orig_text = orig_text
        self.pos_template = pos_template
        self.repitition_template = repitition_template

    def set_props(self, num_lines,num_tokens,num_rep_groups,num_tot_reps,fig_type):
        self.num_lines = num_lines
        self.num_tokens = num_tokens
        self.num_rep_groups = num_rep_groups
        self.num_tot_reps = num_tot_reps
        self.fig_type = fig_type

    def set_orig_rep_words(self, orig_rep_words):
        self.orig_rep_words = orig_rep_words

    def get_orig_rep_words(self):
        return self.orig_rep_words

    def get_unique_id(self):
        return self.unique_id

    def get_orig_text(self):
        return self.orig_text

    def get_pos_template(self):
        return self.pos_template

    def get_repitition_template(self):
        return self.repitition_template

    def get_num_lines(self):
        return self.num_lines

    def get_num_tokens(self):
        return self.num_tokens

    def get_num_rep_groups(self):
        return self.num_rep_groups

    def get_num_tot_reps(self):
        return self.num_tot_reps

    def get_fig_type(self):
        return self.fig_type

    def get_fig_desc(self):
        return FIGURE_DESCRIPTION[self.get_fig_type()].value

    def to_string(self):
        pp = pprint.PrettyPrinter(indent=4, width=160)
        return pprint.pformat(self.__dict__) + "\n"

    def print(self):
        pp = pprint.PrettyPrinter(indent=4, width=160)
        pp.pprint(self.__dict__)
        print("\n")




fig_descs =     [
        "Repetition of the ending word or phrase from the previous clause at the beginning of the next.",
        "Repetition of a word or phrase at the beginning of successive phrases or clauses.",
        "Repetition of words in reverse grammatical order.",
        "The repetition of a word or phrase.",
        "Repetition at the end of a clause of the word or phrase that began it.",
        "Repetition of the same word or phrase at the end of successive clauses.",
        "Repetition of a word or phrase with no others between.",
        "The repetition of word in a short span of text for rhetorical emphasis.",
        "\"Excessive\" repetition of conjunctions between clauses.",
        "Repetition of a word or phrase at the beginning, and of another at the end, of successive clauses."
    ]

rep_figs = ["sonnet_anadiplosis.csv", "sonnet_anaphora.csv", "sonnet_antimetabole.csv",
"sonnet_conduplicatio.csv",
            "sonnet_epanalepsis.csv", "sonnet_epistrophe.csv", "sonnet_epizeuxis.csv",
"sonnet_ploce.csv",
            "sonnet_polysyndeton.csv", "sonnet_symploce.csv"]
unique_id = 0
fig_dict = dict()
for nfig in range(0,num_figs):
    cur_fig_name = fig_names[nfig]
    cur_fig_desc = fig_descs[nfig]
    cur_fig_rhet_file = rep_figs[nfig]

    rhet_table = pd.read_csv(base_path+cur_fig_rhet_file)
    #print(rhet_table)
    fig_ids = rhet_table.figure_id.unique()
    #print(fig_ids)
    for id in fig_ids:
       figure = rhet_table.loc[rhet_table['figure_id'] == id]
       rep_word = []
       sentence_ids = figure['sentence_id'].values
       sentence_ids = uniq(sentence_ids)
       if(not sorted(sentence_ids) == list(range(min(sentence_ids), max(sentence_ids)+1))): continue

       fig_words = uniq([x.lower().replace('''\'d''', '''ed''') for x in list(figure['word'].values)])
       fig_words_dict = dict([[fig_words[y-1], y] for y in range(1, len(fig_words)+1)])
       #print(fig_words_dict)

       orig_text = [ nltk.word_tokenize(sent_table.iloc[i]['text']) for i in sentence_ids]

       #for l in range(0,len(roman)):
           #del orig_text[roman[l][0]][roman[l][1]]

       orig_text = [[orig_text[i][j].lower() for j in range(0,len(orig_text[i]))] for i in range(0,len(orig_text))]
       if(recursive_len(orig_text) > 1):
           tagged = [nltk.pos_tag(orig_text[i]) for i in  range(0,len(orig_text))]
           #print(orig_text)
           pos_templates = [[tagged[i][j][1] for j in range(0,len(tagged[i]))] for i in range(0,len(tagged))]
           #print(pos_templates)

           assert(np.shape(orig_text) == np.shape(pos_templates))
           fig_words_count_dict = dict([[fig_words[y], 0] for y in range(0, len(fig_words))])

           rep_templates = [['0' for j in range(0,len(orig_text[i]))] for i in range(0,len(orig_text))]
           for j in range(0,len(orig_text)):
               for k in range(0,len(orig_text[j])):
                   wordc = orig_text[j][k]
                   for word_rep in list(fig_words_dict.keys()):
                       if(word_rep in wordc):
                           fig_words_count_dict[word_rep] = fig_words_count_dict[word_rep]+1
                           rep_templates[j][k] = str(fig_words_dict[word_rep]) + str(fig_words_count_dict[word_rep])

           unique_id = unique_id+1
           #print("unique_id: " + str(unique_id))
           #print("num_lines: " + str(len(sentence_ids)))
           #print("num_tokens: " + str(recursive_len(orig_text)))
           #print("num_rep_groups: " + str(len(fig_words_dict.keys())))
           #print("num_tot_reps: " + str(sum(fig_words_count_dict.values())))
           #print("fig_type: " + str(RHETORICAL_FIGURES[cur_fig_name.upper()].name)+ "\n")
           #print("Orig. Text:\n")
           orig_line = ", ".join([' '.join(orig_text[i]) for i in range(0,len(orig_text))])
           fig_info = FigureInfo(unique_id, [[sent_table.iloc[i]['text']] for i in sentence_ids], pos_templates, rep_templates)
           fig_info.set_props(len(sentence_ids), recursive_len(orig_text), len(fig_words_dict.keys()), sum(fig_words_count_dict.values()), RHETORICAL_FIGURES[cur_fig_name.upper()].name)
           fig_info.set_orig_rep_words([k  for  k in fig_words_dict.keys()])
           fig_dict[unique_id] = fig_info
           #print(orig_text)
           #print(rep_templates)
           #print(pos_templates)
           #print("Rep. Words:\n")
           #print(fig_words_dict)
           #print("Rep. Words Count:\n")
           #print(fig_words_count_dict)
           #print("POS Template:\n")
           #print(pos_templates)
           #print("Rep Template:\n")
           #print(rep_templates)


   #for index, row in figure.iterrows():

#print(fig_dict.values)


# In[ ]:


from py_files.Limericks import Limerick_Generate
from datetime import datetime
lg = Limerick_Generate(model_dir='/gpt2/models/345M',model_name='345M')
n=200
num_figs=10
max_fig_length=18
base_path = 'data/rhet/repitition/'
base_path_output = 'data/rhet/output/'
corpus = 'sonnets'

single_line = {k: v for k, v in fig_dict.items() if v.get_num_lines() == 1 }

timestamp = int(time.mktime(datetime.now().timetuple()))
out_name = base_path_output + corpus + "_" + str(timestamp) + ".txt"
f = open(out_name,"w+")

fig_and_desc = [e.name + ": " + d.value for e,d in zip(RHETORICAL_FIGURES,FIGURE_DESCRIPTION)]
fig_and_desc.insert(0, "FIGURE: Description\n")
for fd in fig_and_desc:
    f.write(fd+"\n")

for k, v in single_line.items():
    prompt = lg.enc.encode(' '.join(v.get_orig_text()[0]))
    rep_template = v.get_repitition_template()[0]
    pos_template = v.get_pos_template()[0]
    banned_set = v.get_orig_rep_words()

    try:
        f.write(v.to_string())
        new_sentences = lg.gen_line_gpt_cc(cctemplate=rep_template, w=None, encodes=prompt, default_template=pos_template, banned_set=banned_set,search_space=n)
        sents = [new_sentences[j][0] for j in range(len(new_sentences))]
        sent_done = [' '.join(sents[j]) for j in range(len(new_sentences))]
        f.write(" \n".join(sent_done))
        f.flush()
        print(" \n".join(sent_done))

    except Exception:
        print(Exception)
        pass
    f.write("\n-----------------------------------------------------------------------------------------------------------------------------------------------------\n")


# In[ ]:
