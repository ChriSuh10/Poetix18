#### generate the syllabus distribution of all POS

import pickle
from collections import defaultdict, Counter
import numpy as np
import pdb
'''
def create_syll_dict(syllables_file):
    with open(syllables_file, encoding='UTF-8') as f:
        lines = [line.rstrip("\n").split() for line in f if (";;;" not in line)]
        dict_meters = {}
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
            if(newLine[0] not in dict_meters): #THIS IF STATEMENT ALLOWS FOR MULTIPLE PRONUNCIATIONS OF A WORD
                dict_meters[newLine[0]]=[chars]
            else:
                if(chars not in dict_meters[newLine[0]]):
                    dict_meters[newLine[0]]+=[chars]
        dict_meters[','] = ['']
        dict_meters['.'] = ['']
        return dict_meters
'''
if __name__ == '__main__':
	'''
	with open('saved_objects/pos_sylls_mode.p',"rb") as pickle_in:
		pos_sylls_mode=pickle.load(pickle_in)
		pos_sylls_mode["WHILE"]=[(1, 1.0)]
	with open('saved_objects/pos_sylls_mode.p',"wb") as pickle_in:
		pickle.dump(pos_sylls_mode,pickle_in)
	with open('saved_objects/pos_sylls_mode.p',"rb") as pickle_in:
		pos_sylls_mode=pickle.load(pickle_in)
	print("WHILE" in pos_sylls_mode.keys())
	print(pos_sylls_mode)
	'''
	with open('saved_objects/postag_dict_all.p',"rb") as pickle_in:
		postag_dict=pickle.load(pickle_in)
		pos_to_words=postag_dict[1]
		pos_to_words["WHILE"]=['while']
		postag_dict[1]=pos_to_words
	with open('saved_objects/postag_dict_all.p',"wb") as pickle_in:
		pickle.dump(postag_dict,pickle_in)
	with open('saved_objects/postag_dict_all.p',"rb") as pickle_in:
		postag_dict=pickle.load(pickle_in)
		pos_to_words=postag_dict[1]
	print("WHILE" in pos_to_words.keys())
	'''
	syllables_file='saved_objects/cmudict-0.7b.txt'
	postag_file='saved_objects/postag_dict_all.p'
	with open(postag_file, 'rb') as f:
			postag_dict = pickle.load(f)
			pos_to_words = postag_dict[1]
	dict_meters=create_syll_dict(syllables_file)
	
	pos_sylls_mean=defaultdict(float)
	pos_sylls_mode=defaultdict(list)
	pos_sylls_all=defaultdict(int)
	for pos in pos_to_words.keys():
		temp=[]
		total=len(pos_to_words[pos])
		for w in pos_to_words[pos]:
			try:
				temp.append(len(dict_meters[w][0]))
			except:
				total-=1
		if total>0:
			pos_sylls_mean[pos]=np.mean(np.array(temp))
			pos_sylls_mode[pos]=Counter(temp).most_common(5)
			pos_sylls_all[pos]=total
		else:
			pos_sylls_mean[pos]=0
			pos_sylls_mode[pos]=0
	pos_sylls_mode_new={}
	for pos in pos_sylls_mode.keys():
		if pos_sylls_mode[pos]!=0:
			temp=[]
			for i in pos_sylls_mode[pos]:
				temp.append((i[0],i[1]/pos_sylls_all[pos]))
			pos_sylls_mode_new[pos]=temp
		else:
			pos_sylls_mode_new[pos]=[(0,1.0)]

	with open('saved_objects/pos_sylls_mean.p',"wb") as pickle_in:
		pickle.dump(pos_sylls_mean,pickle_in)
	with open('saved_objects/pos_sylls_mode.p',"wb") as pickle_in:
		pickle.dump(pos_sylls_mode_new,pickle_in)
	print(pos_sylls_mean)
	print(pos_sylls_mode)
	print("======================================")
	print(pos_sylls_mode_new)
	
	print("," in pos_to_words.keys())
	print("." in pos_to_words.keys())
	print("," in dict_meters.keys())
	print("." in dict_meters.keys())
	
	'''
	