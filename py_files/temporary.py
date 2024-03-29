import pickle
from collections import defaultdict, Counter
import numpy as np
import pdb
import time
import multiprocessing as mp
import math
from Finer_POS import get_finer_pos_words
import string
import os
import random
from templates import get_new_last_line_templates, get_new_second_line_templates, get_new_third_line_templates, get_new_fourth_line_templates
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
def printing(data, f, f_final, word_embedding_coefficient, template_to_line):
	try:
		with open(f_final+".pickle","rb") as pickle_in:
			data_old=pickle.load(pickle_in)
	except:
		with open(f_final+".pickle","wb") as pickle_in:
			data_old={"score":[],"adjusted_score":[]}
			pickle.dump(data_old,pickle_in)
	data_curr_score=[]
	data_curr_adjusted_score=[]
	temp_data=defaultdict(list)
	for line in data:
		temp_data[" ".join(line[3])].append(line)

	for t,k in enumerate(temp_data.keys()):
		lines=[]
		num_of_words_each_line=[0]
		for pp in temp_data[k]:
			count=0
			for ppp in pp[3]:
				if ppp=="\n":
					count+=1
					num_of_words_each_line.append(0)
				else:
					num_of_words_each_line[count]+=1
			break
		num_of_words_each_line=num_of_words_each_line[1:-1]
		for i in k.split("\n")[1:]:
			i=i.strip()
			if len(i)!=0:
				i_list=i.split(" ")
				try:
					line=list(template_to_line[" ".join(i_list)][0])+["\n"]
				except:
					line=list(template_to_line[" ".join(i_list[:-1])][0])+["\n"]
				lines+=line

		f.write("======================= template: {} ============================  \n".format(t+1))
		f.write(k)
		f.write("----------------------- original sentences ------------------------------------ \n")
		f.write(" ".join(lines))
		for j in temp_data[k]:
			adjusted_score=np.mean(j[1])+word_embedding_coefficient*np.mean(j[5])
			score=np.mean(j[1])
			data_curr_score.append(score)
			data_curr_adjusted_score.append(adjusted_score)
			f.write("-------------------------score:  {};  adjusted_score: {}----------------------- \n".format(score, adjusted_score))
			f.write(" ".join(j[2]))
			f.write("------------------------- score breakdown ------------------------ \n")
			count_w=j[2].index("\n")+1
			count_s=1
			for s in range(4):
				temp_list=[]
				for ww,w in enumerate(j[2][count_w:count_w+num_of_words_each_line[s]]):
					f.write("({} {:03.2f})".format(w,j[1][count_s+ww]))
					temp_list.append(j[1][count_s+ww])
				count_s+=ww
				count_w+=ww+2
				f.write(" line score is : {:04.03f}, look ahead score is : {:04.03f}".format(np.mean(temp_list),j[5][s]))
				f.write("\n")
	data_old_score=data_old["score"]
	data_old_adjusted_score=data_old["adjusted_score"]
	data_curr_score+=data_old_score
	data_curr_adjusted_score+=data_old_adjusted_score
	data_curr={}
	data_curr["score"]=data_curr_score
	data_curr["adjusted_score"]=data_curr_adjusted_score
	with open(f_final+".pickle","wb") as pickle_in:
		pickle.dump(data_curr,pickle_in)
def create_w1s_rhyme_dict(self,names_rhymes_list):
	sum_rhyme=[]
	w1s_rhyme_dict=defaultdict(list)
	words_to_names_rhyme_dict=defaultdict(list)
	for item in names_rhymes_list:
		item_name, item_rhyme= item[0],item[1]
		sum_rhyme+=item_rhyme
	storyline_second_words=sum_rhyme
	print(storyline_second_words)
	for item in names_rhymes_list:
		item_name, item_rhyme= item[0],item[1] 
		for i in item_rhyme:
			if i in storyline_second_words:
				temp=[t for t in item_rhyme if t in storyline_second_words]
				temp.remove(i)
				w1s_rhyme_dict[i]+=temp
				words_to_names_rhyme_dict[i]+=item_name
	return words_to_names_rhyme_dict, w1s_rhyme_dict

def random_split(data,percent=0.5):
    ret=defaultdict(list)
    for i in data.keys():
        ret[i]=[]
        for j in data[i]:
            if random.uniform(0,1)>=0.5:
                ret[i].append(j)
        if len(ret[i])==0:
            del ret[i]
    return ret
def count(data):
	count=0
	for i in data.keys():
		count+=len(data[i])
	return count


if __name__ == '__main__':
	'''
	prompt_list=['surprise', 'humiliation', 'birthday', 'pillow', 'dawn', 'traffic', 'musuem', 'mountain', 'river', 'mud', 'spider', 'rain', 'winter', 'throne', 'beach', 'bank', 'cunning', 'dog', 'blood', 'war', 'world', 'planet', 'fire', 'water', 'violent', 'noble', 'funeral', 'exercise', 'gun', 'music', 'scary', 'creativity', 'evil', 'pride', 'rich', 'leader', 'loss', 'home']
	f = open("success.pickle", "rb")
	data=pickle.load(f)
	temp_2=[p for p in prompt_list if p not in data]
	print(temp_2)
	print(len(temp_2))
	'''
	'''
	with open("saved_objects/unified_poems.pickle","rb") as pickle_in:
		data=pickle.load(pickle_in)
		data1=random.choice(data)
		print(data1)
	'''
	
	with open("saved_objects/templates_processed_tuple.pickle","rb") as pickle_in:
		data=pickle.load(pickle_in)
	second=get_new_second_line_templates()
	third=get_new_third_line_templates()
	fourth=get_new_fourth_line_templates()
	print(count(data["second"]))
	print(count(data["third"]))
	print(count(data["fourth"]))
	### second
	for i in second:
		temp=(tuple(i[0]),tuple(i[1]),(len(i[0]),len(i[0])))
		if i[0][-1] in data["second"].keys():
			data["second"][i[0][-1]].append(temp)
		else:
			data["second"][i[0][-1]]=[]
			data["second"][i[0][-1]].append(temp)
	for i in third:
		temp=(tuple(i[0]),tuple(i[1]),(len(i[0]),len(i[0])))
		if i[0][-1] in data["third"].keys():
			data["third"][i[0][-1]].append(temp)
		else:
			data["third"][i[0][-1]]=[]
			data["third"][i[0][-1]].append(temp)
	for i in fourth:
		temp=(tuple(i[0]),tuple(i[1]),(len(i[0]),len(i[0])))
		if i[0][-1] in data["fourth"].keys():
			data["fourth"][i[0][-1]].append(temp)
		else:
			data["fourth"][i[0][-1]]=[]
			data["fourth"][i[0][-1]].append(temp)

	with open("saved_objects/templates_processed_tuple.pickle","wb") as pickle_in:
		pickle.dump(data,pickle_in)


	

	'''
	with open("saved_objects//blood_100_3_0.1_multi_True_original.pickle","rb") as pickle_in:
		data=pickle.load(pickle_in)
	with open("saved_objects/templates_processed_more_tuple.pickle","rb") as pickle_in:
		templates= pickle.load(pickle_in)
		template_to_line=defaultdict(list)
		for i in ["second","third","fourth","fifth"]:
			for j in templates[i].keys():
				for k in templates[i][j]:
					if k[0][0]=="PRP$" and i=="third":print(" ".join(k[0]))
					template_to_line[" ".join(k[0])].append(k[1])
	saved_directory="testing"
	f_final=saved_directory+"/"+"something"
	if saved_directory not in os.listdir(os.getcwd()):
			os.mkdir(saved_directory)
	with open(saved_directory+"/"+"testting.txt","w") as f:
		#data=[((37437, 323, 508, 2727, 257, 649, 995, 1123, 1110, 13, 383, 1621, 286, 607, 1918, 11, 673, 373, 2923, 416, 257, 582, 11, 673, 373, 1498, 284, 766, 290, 284, 307, 13), (0, -1.8686583, -7.9279566, -1.7537689, -3.5815325, -2.9823372, -7.3178396, -0.7854198, -1.3464375, -3.2089908, -4.305893, -1.7522461, -1.7720312, -6.1531906, -3.006908, -3.8320954, -2.208183, -2.292202, -0.61561483, -1.3350128, -3.2697363, -3.549174, -3.6850667, -0.94812435, -5.939119, -0.014559363, -3.7359376, -3.4837246, -4.6017675, -3.682684, -4.94672), ('there', 'was', 'a', 'kind', 'woman', 'named', 'sunday', '\n', 'who', 'created', 'a', 'new', 'world', 'each', 'day', '.', '\n', 'the', 'story', 'of', 'her', 'death', ',', '\n', 'she', 'was', 'killed', 'by', 'a', 'man', ',', '\n', 'she', 'was', 'able', 'to', 'see', 'and', 'to', 'be', '.', '\n'), ('sunday', '\n', 'WHO', 'VBD', 'A', 'JJ', 'NN', 'EACH', 'NN', '.', '\n', 'THE', 'NN', 'OF', 'PRP$', 'NN', ',', '\n', 'PRP', 'VBD', 'VBN', 'BY', 'A', 'NN', ',', '\n', 'PRP', 'VBD', 'JJ', 'TO', 'VB', 'AND', 'TO', 'VB', '.', '\n'), ('sunday', 'death'))]
		printing(data,f, f_final, 0.1, template_to_line)
	'''
	'''
	mylist=[0, 3, 8, 10, 19, 23, 25, 37, 42, 43, 44, 49, 50, 51, 54, 66, 70, 71, 74, 77, 80, 85, 86, 87, 88, 92, 93, 97, 100, 101, 102, 103, 112, 114, 115, 118, 122, 129, 131, 133, 134, 137,  139, 140, 141, 143, 150, 155, 160, 163, 166, 167, 170, 171, 172]
	with open("saved_objects/last2_tuple.pickle","rb") as f:
		data=pickle.load(f)
		temp_data=defaultdict(list)
		for j,i in enumerate(data):
			if j not in mylist:
				temp_data[i].append(data[i][0])
	with open("saved_objects/last2_tuple_concise.pickle","wb") as pickle_in:
		pickle.dump(temp_data,pickle_in)
	'''
	'''
	limerick_last_two_line_mapping = defaultdict(list)
	special_words= get_finer_pos_words()
	with open("saved_objects/templates_processed_tuple.pickle","rb") as pickle_in:
		data=pickle.load(pickle_in)
	with open("saved_objects/last2_tuple.pickle","rb") as pickle_in:
		last2_dict=pickle.load(pickle_in)
	temp_data={}
	fourth_line_dict={}
	for k in data.keys():
		temp_line=defaultdict(list)
		for i in data[k].keys():
			for j in data[k][i]:
				temp_j=[]
				flag=False
				if len(j[1])!=len(j[0]): continue
				for w in range(len(j[1])):
					if j[1][w].upper() in special_words:
						temp_j.append(j[1][w].upper())
						if w==len(j[1])-1: flag=True
					else:
						temp_j.append(j[0][w])
				if k=="fourth":
					limerick_last_two_line_mapping[tuple(temp_j)]=[]
					fourth_line_dict[tuple(j[1])]=tuple(temp_j)
				if k=="fifth":
					for s in last2_dict[j[1]]:
						limerick_last_two_line_mapping[fourth_line_dict[s]].append(tuple(temp_j))
				if flag:
					temp_line[j[1][-1].upper()].append((tuple(temp_j),j[1],j[2]))
				else:
					temp_line[i].append((tuple(temp_j),j[1],j[2]))
				#if (tuple(temp_j),j[1],j[2]) != j:
					#temp_line[i].append(j)
		temp_data[k]=temp_line
	print(limerick_last_two_line_mapping)
	'''
	'''
	with open("saved_objects/templates_processed_tuple.pickle","rb") as f:
		data=pickle.load(f)
		count=0
		for j in data["fourth"].keys():
			fourth_list = []
			for k in data["fourth"][j]:
				fourth=tuple(k[1])
				for punctuation in string.punctuation:
					if punctuation in fourth:
						print("bad"+str(fourth))
						count+=1
						break
				else:
					fourth_list.append(k)
			data["fourth"][j] = fourth_list
	with open("saved_objects/templates_processed_tuple.pickle","wb") as f:
		pickle.dump(data,f)
	'''	
	
	'''
	with open(postag_file, 'rb') as f:
		postag_dict = pickle.load(f)
		pos_to_words = postag_dict[1]
		words_to_pos=postag_dict[2]
	enc = get_encoder('345M')


	for i in range(60000):
	'''
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
