import py_files.Rhetoric as Rhetoric
import argparse
import os
import pprint
import nltk
import numpy as np
import time
import pandas as pd
from datetime import datetime
from py_files.Limericks import Limerick_Generate

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


#os.getcwd()
parser = argparse.ArgumentParser(description='Do rhetorical analysis')

parser.add_argument('input', help='''Input file name (located in data/) e.g. sonnets.txt''')


args = parser.parse_args()

local_path_input = 'data/' + args.input
abs_path_input = os.path.abspath(local_path_input)
filename, file_extension = os.path.splitext(abs_path_input)
corpus, extension = os.path.splitext(args.input)

corpus_input = 'data/rhet/' + corpus + '_input_corpus' + '.txt'
sent_table = pd.read_csv(corpus_input, delimiter='|', usecols=['text'])
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

pwd_dir = os.path.dirname(os.path.realpath(__file__))
output_folder_rep = "%s/data/rhet/repitition" % (pwd_dir)

unique_id = 0
fig_dict = dict()

for e,d in zip(Rhetoric.RHETORICAL_FIGURES, Rhetoric.FIGURE_DESCRIPTION):

    cur_fig_name = e.name
    cur_fig_desc = d.value
    cur_fig_rhet_file = '''%s/%s_%s.csv''' % (output_folder_rep, corpus, e.name.lower())
    print(cur_fig_desc)
    print(cur_fig_rhet_file)

    rhet_table = pd.read_csv(cur_fig_rhet_file)
    # print(rhet_table)
    fig_ids = rhet_table.figure_id.unique()
    # print(fig_ids)
    for id in fig_ids:
        figure = rhet_table.loc[rhet_table['figure_id'] == id]
        rep_word = []
        sentence_ids = figure['sentence_id'].values
        sentence_ids = uniq(sentence_ids)
        if (not sorted(sentence_ids) == list(range(min(sentence_ids), max(sentence_ids) + 1))): continue

        fig_words = uniq([x.lower().replace('''\'d''', '''ed''') for x in list(figure['word'].values)])
        fig_words_dict = dict([[fig_words[y - 1], y] for y in range(1, len(fig_words) + 1)])
        # print(fig_words_dict)

        orig_text = [nltk.word_tokenize(sent_table.iloc[i]['text']) for i in sentence_ids]

        # for l in range(0,len(roman)):
        # del orig_text[roman[l][0]][roman[l][1]]

        orig_text = [[orig_text[i][j].lower() for j in range(0, len(orig_text[i]))] for i in range(0, len(orig_text))]
        if (recursive_len(orig_text) > 1):
            tagged = [nltk.pos_tag(orig_text[i]) for i in range(0, len(orig_text))]
            # print(orig_text)
            pos_templates = [[tagged[i][j][1] for j in range(0, len(tagged[i]))] for i in range(0, len(tagged))]
            # print(pos_templates)

            assert (np.shape(orig_text) == np.shape(pos_templates))
            fig_words_count_dict = dict([[fig_words[y], 0] for y in range(0, len(fig_words))])

            rep_templates = [['0' for j in range(0, len(orig_text[i]))] for i in range(0, len(orig_text))]
            for j in range(0, len(orig_text)):
                for k in range(0, len(orig_text[j])):
                    wordc = orig_text[j][k]
                    for word_rep in list(fig_words_dict.keys()):
                        if (word_rep in wordc):
                            fig_words_count_dict[word_rep] = fig_words_count_dict[word_rep] + 1
                            rep_templates[j][k] = str(fig_words_dict[word_rep]) + str(fig_words_count_dict[word_rep])

            unique_id = unique_id + 1
            #print("unique_id: " + str(unique_id))
            #print("num_lines: " + str(len(sentence_ids)))
            #print("num_tokens: " + str(recursive_len(orig_text)))
            #print("num_rep_groups: " + str(len(fig_words_dict.keys())))
            #print("num_tot_reps: " + str(sum(fig_words_count_dict.values())))
            #print("fig_type: " + str(cur_fig_name)+ "\n")
            #print("Orig. Text:\n")
            orig_line = ", ".join([' '.join(orig_text[i]) for i in range(0, len(orig_text))])
            fig_info = Rhetoric.FigureInfo(unique_id, [[sent_table.iloc[i]['text']] for i in sentence_ids], pos_templates,
                                  rep_templates)
            fig_info.set_props(len(sentence_ids), recursive_len(orig_text), len(fig_words_dict.keys()),
                               sum(fig_words_count_dict.values()), e.name)
            fig_info.set_orig_rep_words([k for k in fig_words_dict.keys()])
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

# for index, row in figure.iterrows():

#print(fig_dict.values)
timestamp = int(time.mktime(datetime.now().timetuple()))
lg = Limerick_Generate(model_dir='gpt2/models/345M', model_name='345M')
n = 200
top_sent = 20

for e,d in zip(Rhetoric.RHETORICAL_FIGURES, Rhetoric.FIGURE_DESCRIPTION):

    #out_name = 'data/rhet/output/' + corpus + "_" + e.name.lower() + "_" + str(timestamp)  + ".txt"
    #f = open(out_name,"w+")

    fig_and_desc = e.name + ": " + d.value
    #f.write("FIGURE: Description\n")
    #f.write(fig_and_desc +" \n\n")

    #print(out_name)
    #print(fig_and_desc)
    filtered = {k: v for k, v in fig_dict.items() if(v.get_num_lines() == 2 and v.get_fig_type() == e.name)}
    #print(len(filtered))   


    for k, v in filtered.items():
       #print(fig_and_desc)
    
       try:
           gen_lines = lg.gen_line_gpt_rep_multiline(v, search_space=n, top_sent=top_sent)
           v.set_gen_lines(gen_lines)
           f.write(v.to_string())
           f.write("\n-----------------------------------------------------------------------------------------------------------------------------------------------------\n")   
           f.flush()
        
       except Exception:
          print(Exception)
          pass
