import tensorflow as tf
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords
import collections
from collections import defaultdict
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
import pdb
from .model_back import Model as Model_back
from .functions import search_back_meter
from .templates import get_templates
from gpt2.src.score import score_model
from gpt2.src.generate_prompt import generate_prompt
from gpt2.src.encoder import get_encoder
from .templates import get_first_nnp, get_first_line_templates
import pickle
from .Limericks import Limerick_Generate
random.seed(20)
class Limerick_Generate_new(Limerick_Generate):
	def __init__(self, wv_file='py_files/saved_objects/poetic_embeddings.300d.txt',
            syllables_file='py_files/saved_objects/cmudict-0.7b.txt',
            postag_file='py_files/saved_objects/postag_dict_all.p',
            model_dir='gpt2/models/345M',
            model_name='345M'):
		super(Limerick_Generate_new,self).__init__()
		self.api_url='https://api.datamuse.com/words'
		self.model_dir = model_dir
		self.model_name = model_name
		self.poetic_vectors = KeyedVectors.load_word2vec_format(wv_file, binary=False)
		self.create_syll_dict(syllables_file)
		with open(postag_file, 'rb') as f:
		    postag_dict = pickle.load(f)
		self.pos_to_words = postag_dict[1]
		self.words_to_pos = postag_dict[2]
		self.create_pos_syllables()
		self.create_templates_dict(postag_dict[0])

		self.first_line_words=pickle.load(open('py_files/saved_objects/first_line.p','rb'))
		self.width = 20
		# Not sure what this does, necessary for search_back function
		self.word_pools = [set([]) for n in range(4)]
		self.enc = get_encoder(self.model_name)
		# get male and female names
		with open("py_files/saved_objects/dist.female.first.txt", "r") as hf:
		    self.female_names = [lines.split()[0].lower() for lines in hf.readlines()]
		with open("py_files/saved_objects/dist.male.first.txt", "r") as hf:
		    self.male_names = [lines.split()[0].lower() for lines in hf.readlines()]
		with open("py_files/saved_objects/templates_punctuation.pickle","rb") as pickle_in:
			self.templates= pickle.load(pickle_in)
		with open("py_files/saved_objects/pos_sylls_mode.p","rb") as pickle_in:
			self.pos_sylls_mode= pickle.load(pickle_in)
	def gen_poem_andre_new(self,prompt,search_space,thresh_hold):
		w1s_rhyme_dict, w3s_rhyme_dict= self.get_two_sets_henry(prompt)
		self.w1s_rhyme_dict=w1s_rhyme_dict
		self.w3s_rhyme_dict=w3s_rhyme_dict
		female_name_list, male_name_list=self.load_name_list()
		for name in w1s_rhyme_dict.keys():
			if name.lower() not in female_name_list and  name.lower() not in male_name_list:
				del w1s_rhyme_dict[name]
		assert len(w1s_rhyme_dict.keys())>0, "no storyline available"
		last_word_dict=self.last_word_dict(w1s_rhyme_dict,w3s_rhyme_dict)
		with open("limericks_data_new/"+prompt+"_"+str(search_space)+"_"+str(thresh_hold)+".txt","a+") as f:
			previous_data=[]
			for i in w1s_rhyme_dict.keys():
				f.write("================================ 125 rhymes ===================================")
				f.write(i+":"+"\n")
				f.write(" ".join(w1s_rhyme_dict[i])+"\n")
				text=random.choice(self.gen_first_line_new(i.lower(),strict=True))
				first_line_encodes = self.enc.encode(" ".join(text))
				# previous data [(encodes,score, text, template, (w1,w3))]
				previous_data.append((first_line_encodes,0,text+["\n"], [text[-1],"\n"],(i,"")))
			for i in w3s_rhyme_dict.keys():
				f.write("=============================== 34 rhymes  =====================================")
				f.write(i+":"+"\n")
				f.write(" ".join(w3s_rhyme_dict[i])+"\n")
			for which_line, num_sylls in zip(["second","third","fourth","fifth"],[9,6,6,9]):
			#for which_line, num_sylls in zip(["fifth"],[9]):
				print("======================= starting {} line generation =============================".format(which_line))
				last_word_set=last_word_dict[which_line]
				possible=self.get_all_templates(num_sylls,which_line,last_word_set)
				previous_data=self.gen_line_flexible(previous_data=previous_data, possible=possible,num_sylls=num_sylls, search_space=search_space, thresh_hold=thresh_hold, which_line=which_line)
			temp_data=defaultdict(list)
			for i in previous_data:
				temp_data[" ".join(i[3])].append(i)
			for i,k in enumerate(temp_data.keys()):
				f.write("=======================   template: {}   ============================  \n".format(i+1))
				f.write(k+"\n")
				for j in temp_data[k]:
					f.write("-------------------------  score:  {}    ----------------------- \n".format(j[1]/len(j[3])))
					f.write(" ".join(j[2]))
				f.write("=====================================================  \n")




	def encodes_align(self,previous_data):
		encodes_length=[len(i[0]) for i in previous_data]
		encodes=[i[0][-min(encodes_length):] for i in previous_data]
		temp=[]
		for i,j in enumerate(previous_data):
			temp.append((encodes[i],j[1],j[2],j[3],j[4]))
		return temp
	def last_word_dict(self, w1s_rhyme_dict,w3s_rhyme_dict):
		last_word_dict={}
		for i in ["second","third","fourth","fifth"]:
			temp=[]
			if i == "second" or i =="fifth":
				for k in w1s_rhyme_dict.keys():
					temp+=w1s_rhyme_dict[k]
			if i== "third" or i== "fourth":
				for k in w3s_rhyme_dict.keys():
					temp+=w3s_rhyme_dict[k]
			last_word_dict[i]=list(set(temp))
		return last_word_dict
	def sylls_bounds(self,partial_template):
		threshold=0.1
		sylls_up=0
		sylls_lo=0
		for t in partial_template:
			x=[j[0] for j in self.pos_sylls_mode[t] if j[1]>=min(threshold,self.pos_sylls_mode[t][0][1])]
			if len(x)==0:
				sylls_up+=0
				sylls_lo+=0
			else:
				sylls_up+=max(x)
				sylls_lo+=min(x)
		return sylls_up, sylls_lo
	def there_is_template_new(self,last_word_info,num_sylls, which_line):
		# return a list of possible templates
		threshold=0.1
		pos=last_word_info[0]
		sylls=last_word_info[1]
		dataset=self.templates[which_line]
		possible=[]
		if pos in dataset.keys():
			for i,_,_ in dataset[pos]:
				sylls_up=0
				sylls_lo=0
				for t in i[:-1]:
					x=[j[0] for j in self.pos_sylls_mode[t] if j[1]>=min(threshold,self.pos_sylls_mode[t][0][1])]
					sylls_up+=max(x)
					sylls_lo+=min(x)
				if num_sylls-sylls>=sylls_lo and num_sylls-sylls<=sylls_up:
					possible.append(i)
		return possible
	def unique_list(self,x):
		output=[]
		for i in x:
			if i not in output:
				output.append(i)
		return output

	def get_all_templates(self,num_sylls,which_line, last_word_set):
		last_word_info_set=set()
		temp=[]
		for i in last_word_set:
			if i in self.words_to_pos.keys() and i in self.dict_meters.keys():
				for j in self.words_to_pos[i]:
					last_word_info_set.add((j,len(self.dict_meters[i][0])))
		for i in last_word_info_set:
			temp+=self.there_is_template_new(i, num_sylls, which_line)
		temp=self.unique_list(temp)
		return temp
	def template_sylls_checking(self,pos_set,sylls_set,template_curr,num_sylls_curr,possible, num_sylls):
		continue_flag=[]
		for t in possible:
			if t[:len(template_curr)]==template_curr and len(t)>len(template_curr)+1:
				for pos in pos_set:
					if pos==t[len(template_curr)]:
						for sylls in sylls_set:
							sylls_up, sylls_lo=self.sylls_bounds(t[len(template_curr)+1:])
							if num_sylls-num_sylls_curr-sylls>=sylls_lo and num_sylls-num_sylls_curr-sylls<=sylls_up:
								continue_flag.append((pos,sylls))
		if len(continue_flag)==0: continue_flag=False
		return continue_flag
	def end_template_checking(self,pos_set,sylls_set,template_curr,num_sylls_curr,possible, num_sylls, debug=False):
		end_flag=[]
		if debug: pdb.set_trace()
		for t in possible:
			if t[:len(template_curr)]==template_curr and len(t)==len(template_curr)+1:
				if t[len(template_curr)] in [",","."] and list(pos_set)[0]==t[len(template_curr)]:
					end_flag.append((list(pos_set)[0],list(sylls_set)[0]))
				else:
					for pos in pos_set:
						if pos==t[len(template_curr)]:
							for sylls in sylls_set:
								if num_sylls_curr+sylls==num_sylls:
									end_flag.append((pos,sylls))
		if len(end_flag)==0:
			end_flag=False
		return end_flag
	def diversity_sort(self,search_space, data, finished):
		temp_data=defaultdict(list)
		for n in data:
			if not finished:
				key=";".join(n[3]+n[4])
			else:
				key=";".join(n[3])
			temp_data[key].append(n)
		data=[]
		break_point=0
		list_of_keys=list(temp_data.keys())
		x=random.sample(list_of_keys, len(list_of_keys))
		for k in x:
			if not finished:
				temp=heapq.nlargest(1, temp_data[k], key=lambda x: x[1]/(len(x[3])+len(x[4])))
			else:
				temp=heapq.nlargest(1, temp_data[k], key=lambda x: x[1]/len(x[3]))
			data+=temp
			break_point+=1
			#if break_point>=20: break
		return random.sample(data, min(len(data),search_space)), len(temp_data.keys())

	def gen_line_flexible(self, previous_data, possible,num_sylls, search_space, thresh_hold, which_line):
		'''
		threshold=10, 10 best words that satisfy constraints that are not the last word.
		'''
		previous_data=self.encodes_align(previous_data)
		sentences=[]
		for i in previous_data:
			template_curr=[]
			num_sylls_curr=0
			sentences.append([i[0],i[1],i[2],i[3],template_curr,num_sylls_curr,i[4]])
		finished_sentences=[]
		iteration=0
		new_sentences=[1]
		while(len(new_sentences)>0):
			iteration+=1
			context_token=[s[0] for s in sentences]
			m=len(context_token)
			context_token=np.array(context_token).reshape(m,-1)
			logits = score_model(model_name=self.model_name, context_token = context_token)
			new_sentences=[]
			for i,j in enumerate(logits):
				sorted_index=np.argsort(-1*j)
				break_point_continue=0
				break_point_end=0
				for index in sorted_index:
					word = self.enc.decode([index]).lower().strip()
					if len(word) == 0:
						continue
					# note that both , and . are in these keys()
					elif word not in self.words_to_pos.keys() or word not in self.dict_meters.keys():
						continue
					else:
						pos_set=set(self.words_to_pos[word])
						sylls_set=set([len(m) for m in self.dict_meters[word]])
						if len(pos_set)==0 or len(sylls_set==0):
							print(word)
							continue
						template_curr=sentences[i][4]
						num_sylls_curr=sentences[i][5]
						# end_flag is the (POS, Sylls) of word if word can be the last_word for a template, False if not
						# continue_flag is (POS,Sylls) if word can be in a template and is not the last word. False if not
						debug=False
						continue_flag=self.template_sylls_checking(pos_set=pos_set,sylls_set=sylls_set,template_curr=template_curr,num_sylls_curr=num_sylls_curr,possible=possible, num_sylls=num_sylls)
						end_flag=self.end_template_checking(pos_set=pos_set,sylls_set=sylls_set,template_curr=template_curr,num_sylls_curr=num_sylls_curr,possible=possible, num_sylls=num_sylls, debug=debug)
						if continue_flag:
							for continue_sub_flag in continue_flag:
								new_sentences.append([sentences[i][0] + [index],
													sentences[i][1] + np.log(j[index]),
													sentences[i][2]+[word],
													sentences[i][3],
													sentences[i][4]+[continue_sub_flag[0]],
													sentences[i][5]+continue_sub_flag[1],
													sentences[i][6]])
								break_point_continue+=1
						if end_flag:
							for end_sub_flag in end_flag:
								if which_line=="second" or which_line=="fifth":
									if word in self.w1s_rhyme_dict[sentences[i][6][0]]:
										finished_sentences.append([sentences[i][0] + [index],
													sentences[i][1] + np.log(j[index]),
													sentences[i][2]+[word],
													sentences[i][3]+sentences[i][4]+[end_sub_flag[0]],
													sentences[i][6]])
										break_point_end+=1
								if which_line=="third":
									if word in self.w3s_rhyme_dict.keys():
										finished_sentences.append([sentences[i][0] + [index],
													sentences[i][1] + np.log(j[index]),
													sentences[i][2]+[word],
													sentences[i][3]+sentences[i][4]+[end_sub_flag[0]],
													(sentences[i][6][0],word)])
										break_point_end+=1
								if which_line=="fourth":
									if word in self.w3s_rhyme_dict[sentences[i][6][1]]:
										finished_sentences.append([sentences[i][0] + [index],
													sentences[i][1] + np.log(j[index]),
													sentences[i][2]+[word],
													sentences[i][3]+sentences[i][4]+[end_sub_flag[0]],
													sentences[i][6]])
										break_point_end+=1
			print("========================= iteration {} ends ============================= \n".format(iteration))
			sentences, diversity=self.diversity_sort(search_space,new_sentences, finished=False)
			print("{} sentences before diversity_sort, {} sentences afterwards, diversity {}, now {} finished_sentences".format(len(new_sentences),len(sentences), diversity, len(finished_sentences)))
			'''
			for sen in sentences:
				print(sen)
				print("\n")
			'''
		assert len(sentences)==0, "something wrong"
		previous_data_temp, _=self.diversity_sort(search_space,finished_sentences, finished=True)
		previous_data=[(i[0],i[1],i[2]+["\n"],i[3]+["\n"],i[4]) for i in previous_data_temp]
		return previous_data

