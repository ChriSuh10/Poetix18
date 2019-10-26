import pickle
from collections import defaultdict, Counter
import numpy as np
import pdb
import time
import multiprocessing as mp
import math
from gpt2.src.encoder import get_encoder
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
def f(a_list):
    out = 0
    for n in a_list:
        out += n*n
        time.sleep(0.1)
 
    return out
 
def f_mp(a_list):
	print(mp.cpu_count())
	chunks = [a_list[i::5] for i in range(10)]
	pool = mp.Pool(processes=10)
	result = pool.map(f, chunks)
	print(result)
def split_mp(data):
	d=len(data)//4
	temp=[]
	for i in range(4):
		if i!=3:
			temp.append(data[i*d:(i+1)*d])
		else:
			temp.append(data[i*d:])
	return temp
def split_chunks( data):
	data_list=[]
	cpu=mp.cpu_count()
	chuck_len = len(data)//cpu + 1
	flag=0
	while_flag=True
	while (while_flag):
		if flag+chuck_len<len(data):
			data_list.append(data[flag:flag+chuck_len])
		else:
			data_list.append(data[flag:])
			while_flag=False
		flag+=chuck_len
	return data_list
def is_correct_meter(template, num_syllables=[8], stress=[1, 4, 7]):
	template=['there', 'was', 'a', 'young', 'fellow', 'named', 'salvatore']
	meter = []
	n = 0
	for x in template:
	    if x not in dict_meters:
	        return False
	    n += len(dict_meters[x][0])
	    curr_meter = dict_meters[x]
	    for i in range(max([len(j) for j in curr_meter])):
	        curr_stress = []
	        for possible_stress in curr_meter:
	        	if len(possible_stress)>=i+1:
	        		curr_stress.append(possible_stress[i])
	        meter.append(curr_stress)
	return (not all(('1' not in meter[i]) for i in stress)) and (n in num_syllables)

if __name__ == '__main__':
	
	syllables_file='saved_objects/cmudict-0.7b.txt'
	postag_file='saved_objects/postag_dict_all.p'
	dict_meters=create_syll_dict(syllables_file)
	with open(postag_file, 'rb') as f:
		postag_dict = pickle.load(f)
		pos_to_words = postag_dict[1]
		words_to_pos=postag_dict[2]
	enc = get_encoder('345M')


	for i in range(60000):

	'''
	special_pos="in dt wdt wp md cc cd ex pdt wrb rp wp$"
	special_pos=[i.upper() for i in special_pos.split(" ")]
	special_words=set()
	for k in special_pos:
		for j in pos_to_words[k]:
			special_words.add(j.upper())
	'''
	'''
	with open("saved_objects/pos_sylls_mode.p","rb") as pickle_in:
		pos_sylls_mode= pickle.load(pickle_in)
	for i in special_words:
		try:
			pos_sylls_mode[i]=[(len(dict_meters[i.lower()][0]),1.0)]
		except:
			pos_sylls_mode[i]=[1,1.0]
	for i in pos_sylls_mode.keys():
		print("{}:{}".format(i, pos_sylls_mode[i]))
	'''
		
	'''
	with open("saved_objects/templates_processed_more_tuple.pickle","rb") as pickle_in:
		templates= pickle.load(pickle_in)
	for i in templates["second"]:
		for j in templates["second"][i]:
			print(j)
			print("\n")
	'''

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
	
	syllables_file='saved_objects/cmudict-0.7b.txt'
	postag_file='saved_objects/postag_dict_all.p'
	dict_meters=create_syll_dict(syllables_file)
	with open(postag_file, 'rb') as f:
		postag_dict = pickle.load(f)
		pos_to_words = postag_dict[1]
	special_pos="in dt wdt wp md cc cd ex pdt wrb rp wp$"
	special_pos=[i.upper() for i in special_pos.split(" ")]
	special=set()
	for k in special_pos:
		for j in pos_to_words[k]:
			special.add(j.upper())
	print(special)

	with open("saved_objects/templates_processed_tuple.pickle","rb") as pickle_in:
		data=pickle.load(pickle_in)
	temp_data={}
	for k in data.keys():
		temp_line=defaultdict(list)
		for i in data[k].keys():
			for j in data[k][i]:
				temp_j=[]
				flag=False
				if len(j[1])!=len(j[0]): continue
				for w in range(len(j[1])):
					if j[1][w].upper() in special:
						temp_j.append(j[1][w].upper())
						if w==len(j[1])-1: flag=True
					else:
						temp_j.append(j[0][w])
				if flag: 
					temp_line[j[1][-1].upper()].append((tuple(temp_j),j[1],j[2]))
				else:
					temp_line[i].append((tuple(temp_j),j[1],j[2]))
				#if (tuple(temp_j),j[1],j[2]) != j:
					#temp_line[i].append(j)
		temp_data[k]=temp_line


	with open("saved_objects/templates_processed_more_tuple.pickle","wb") as pickle_in:
		pickle.dump(temp_data,pickle_in)
	
	
	with open("saved_objects/templates_processed_more_tuple.pickle","rb") as pickle_in:
		data=pickle.load(pickle_in)
		
		for k in data.keys():
			print(k)
			for j in data[k].keys():
				print(j)
				for i in data[k][j]:
					print(i)
 
	postag_file='saved_objects/postag_dict_all.p'
	syllables_file='saved_objects/cmudict-0.7b.txt'
	dict_meters=create_syll_dict(syllables_file)
	with open(postag_file,"rb") as f:
		postag_dict = pickle.load(f)
		words_to_pos = postag_dict[2]
	word="."
	pos_set=set(words_to_pos[word])
	sylls_set=set([len(m) for m in dict_meters[word]])
	print(pos_set)
	print(sylls_set)
	'''		
	
	'''
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
			
	
	'''
	with open("saved_objects/templates_new3.pickle","rb") as pickle_in:
		data=pickle.load(pickle_in)
	wu=data["second"]
	for i in wu.keys():
		for j in wu[i]:
			print(j)
	print("==============================================================")
	print("==============================================================")
	with open("saved_objects/templates_punctuation.pickle","rb") as pickle_in:
		data=pickle.load(pickle_in)
	wu=data["second"]
	for i in wu.keys():
		for j in wu[i]:
			print(j)
	'''
	'''
	with open("saved_objects/templates_processed_more_tuple.pickle","rb") as pickle_in:
		data=pickle.load(pickle_in)
	for i in ["second","third","fourth","fifth"]:
		print("======================= \n")
		for j in data[i].keys():
			for k in data[i][j]:
				print(k)
	'''
	'''
	with open("saved_objects/template_to_line.pickle","wb") as pickle_in:
		pickle.dump(template_to_line,pickle_in)
	'''
		
'''
	with open("saved_objects/template_to_line.pickle","rb") as pickle_in:
		data=pickle.load(pickle_in)
		print(data)
'''
'''

	with open("saved_objects/template_to_line.pickle","rb") as pickle_in:
		template_to_line=pickle.load(pickle_in)
	temp_data={'carlene \n CC PRP$ NN , VBN NNP , \n':0}
	for i,k in enumerate(temp_data.keys()):
		lines=[]
		for i in k.split("\n")[1:]:
			i=i.strip()
			if len(i)!=0:
				i_list=i.split(" ")
				if i_list[-1] in [",","."]:
					i_list=i_list[:-1]
				line=template_to_line[" ".join(i_list)][0]+["\n"]
				lines+=line
	print(lines)
'''
