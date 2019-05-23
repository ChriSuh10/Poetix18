import pandas as pd
import re
# Open a file: file
sent_table = pd.read_csv('data/corpus.txt', delimiter='|', usecols=['text'])
print(sent_table)


for i in sent_table.index.values:
    print(sent_table.iloc[i]['text']);
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

import nltk
import numpy as np
import time
from datetime import datetime

num_figs=10
max_fig_length=18
base_path = 'data/rhet/repitition/'
base_path_output = 'data/rhet/output/'
corpus = 'sonnets'

fig_names = ["Anadiplosis", "Anaphora", "Antimetabole", "Conduplicatio", "Epanalepsis", "Epistrophe", "Epizeuxis", "Ploce", "Polysyndeton", "Symploce"]

fig_descs = \
    [
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

from py_files.Limericks import Limerick_Generate
lg = Limerick_Generate(model_dir='/gpt2/models/345M',model_name='345M')
n=200
banned_set = ['foo']
for nfig in range(0,num_figs):
    
    cur_fig_name = fig_names[nfig]
    cur_fig_desc = fig_descs[nfig]
    cur_fig_rhet_file = rep_figs[nfig]

    rhet_table = pd.read_csv(base_path+cur_fig_rhet_file)
    timestamp = int(time.mktime(datetime.now().timetuple()))
    out_name = base_path_output + corpus + "_" + cur_fig_name + "_" + str(timestamp) +".txt"
    f = open(out_name,"w+")
    
    f.write("FIGURE: %s\n" % cur_fig_name)
    f.write("DESCRIPTIOM: %s\n" % cur_fig_desc)
    f.write("CORPUS: %s\n\n" % corpus)
    fig_ids = rhet_table.figure_id.unique()
    #print(fig_ids)
    for id in fig_ids:
       figure = rhet_table.loc[rhet_table['figure_id'] == id]
       rep_word = []
       sentence_ids = figure['sentence_id'].values
       sentence_ids = uniq(sentence_ids)
       if(not sorted(sentence_ids) == list(range(min(sentence_ids), max(sentence_ids)+1))): continue 
       print(sentence_ids)
       fig_words = uniq([x.lower().replace('''\'d''', '''ed''') for x in
list(figure['word'].values)])
       fig_words_dict = dict([[fig_words[y-1], y] for y in range(1, len(fig_words)+1)])
       #print(fig_words_dict)
        
       orig_text = [ nltk.word_tokenize(sent_table.iloc[i]['text']) for i in sentence_ids]
       pattern = '^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
       roman = []
       for j in range(0,len(orig_text)):
           for k in range(0,len(orig_text[j])):
               if(re.search(pattern, orig_text[j][k])):
                   roman.append([j,k])
       
       #for l in range(0,len(roman)):
           #del orig_text[roman[l][0]][roman[l][1]]
    
       orig_text = [[orig_text[i][j].lower() for j in range(0,len(orig_text[i]))] for i
in range(0,len(orig_text))]
       if(len(sentence_ids) == 1 and recursive_len(orig_text) <= 18 and recursive_len(orig_text) > 1):
           tagged = [nltk.pos_tag(orig_text[i]) for i in  range(0,len(orig_text))]
           #print(orig_text)
           pos_templates = [[tagged[i][j][1] for j in range(0,len(tagged[i]))] for i in
range(0,len(tagged))]
           #print(pos_templates)
           
           assert(np.shape(orig_text) == np.shape(pos_templates))
           fig_words_count_dict = dict([[fig_words[y], 0] for y in range(0,
len(fig_words))])
        
           rep_templates = [['0' for j in range(0,len(orig_text[i]))] for i in
range(0,len(orig_text))]
           for j in range(0,len(orig_text)):
               for k in range(0,len(orig_text[j])):
                   wordc = orig_text[j][k]
                   for word_rep in list(fig_words_dict.keys()):
                       if(word_rep in wordc):
                           fig_words_count_dict[word_rep] = fig_words_count_dict[word_rep]+1
                           rep_templates[j][k] = str(fig_words_dict[word_rep]) + str(fig_words_count_dict[word_rep])
           
           f.write("Orig. Text:\n")
           orig_line = ", ".join([' '.join(orig_text[i]) for i in
range(0,len(orig_text))])
           print(orig_line)
           f.write(orig_line)
           f.write("\n")
           f.write("Rep. Words:\n")
           f.writelines(', '.join(fig_words_dict))
           f.write("\n")
           f.write("POS Template:\n")
           f.writelines(str(pos_templates))
           f.write("\n")
           f.write("Rep Template:\n")
           f.writelines(str(rep_templates))
           f.write("\n")
           f.write("Sample Output:\n")
           
           print(rep_templates[0])
           print(pos_templates[0])
           
           prompt = lg.enc.encode(orig_line + ",")
           try:
               new_sentences = lg.gen_line_gpt_cc(cctemplate=rep_templates[0], w=None, encodes=prompt, default_template=pos_templates[0], banned_set=banned_set,search_space=n)
               sents = [new_sentences[j][0] for j in range(len(new_sentences))]
               sent_done = [' '.join(sents[j]) for j in range(len(new_sentences))]
               f.write(" \n".join(sent_done))
 
           except Exception:
               pass
           f.write("\n-----------------------------------------------------------------------------------------------------------------------------------------------------\n") 
           print(rep_templates)     
  
   #for index, row in figure.iterrows():
    f.close()
   #source = sentences
   # while(!re.search(exp, source)
        
