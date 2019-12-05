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
from gpt2.src.score import score_model
from gpt2.src.generate_prompt import generate_prompt
from gpt2.src.encoder import get_encoder
from .templates import get_first_nnp, get_first_line_templates
import pickle
from .Limericks import Limerick_Generate
from .Finer_POS import get_finer_pos_words
import multiprocessing as mp
import time
random.seed(20)

class Limerick_Generate_new(Limerick_Generate):
	def __init__(self, wv_file='py_files/saved_objects/poetic_embeddings.300d.txt',
            syllables_file='py_files/saved_objects/cmudict-0.7b.txt',
            postag_file='py_files/saved_objects/postag_dict_all.p',
            model_dir='gpt2/models/345M',
            model_name='345M',
            saved_directory=None):
		print("=================== Initializing ==================================")
		super(Limerick_Generate_new,self).__init__()
		self.saved_directory=saved_directory
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
		self.cpu=mp.cpu_count()
		print(self.cpu)
		self.first_line_words=pickle.load(open('py_files/saved_objects/first_line.p','rb'))
		self.width = 20
		self.enc = get_encoder(self.model_name)
		# get male and female names
		with open("py_files/saved_objects/dist.female.first.txt", "r") as hf:
		    self.female_names = [lines.split()[0].lower() for lines in hf.readlines()]
		with open("py_files/saved_objects/dist.male.first.txt", "r") as hf:
		    self.male_names = [lines.split()[0].lower() for lines in hf.readlines()]
		# punctuations
		self.punctuation={"second":True,"third":True,"fourth":True,"fifth":True}
		self.sentence_to_punctuation={"second":".","third":",","fourth":",","fifth":"."}
		with open(self.filtered_names_rhymes, "rb") as hf:
			self.names_rhymes_list = pickle.load(hf)

		# word embedding coefficients

	def create_w1s_rhyme_dict(self,prompt):
		self.sum_rhyme=[]
		self.w1s_rhyme_dict=defaultdict(list)
		self.words_to_names_rhyme_dict=defaultdict(list)
		for item in self.names_rhymes_list:
			item_name, item_rhyme= item[0],item[1]
			self.sum_rhyme+=item_rhyme
		self.storyline_second_words=self.get_similar_word_henry([prompt], n_return=200, word_set=set(self.sum_rhyme))
		print(self.storyline_second_words)
		for item in self.names_rhymes_list:
			item_name, item_rhyme= item[0],item[1] 
			for i in item_rhyme:
				if i in self.storyline_second_words:
					temp=[t for t in item_rhyme if t in self.storyline_second_words]
					temp.remove(i)
					self.w1s_rhyme_dict[i]+=temp
					self.words_to_names_rhyme_dict[i]+=item_name


	def finer_pos_category(self):
		self.special_words= get_finer_pos_words()
		# special_pos="in dt wdt wp md cc cd ex pdt wrb rp wp$"
		# special_pos=[i.upper() for i in special_pos.split(" ")]
		# for k in special_pos:
		# 	for j in self.pos_to_words[k]:
		# 		self.special_words.add(j.upper())
		if self.mode=="multi":
			with open("py_files/saved_objects/templates_processed_tuple.pickle","rb") as pickle_in:
				data=pickle.load(pickle_in)
		else:
			with open("py_files/saved_objects/unified_poems.pickle","rb") as pickle_in:
				data=pickle.load(pickle_in)
				data=data[self.mode]
		temp_data={}
		for k in data.keys():
			temp_line=defaultdict(list)
			for i in data[k].keys():
				for j in data[k][i]:
					temp_j=[]
					flag=False
					if len(j[1])!=len(j[0]): continue
					for w in range(len(j[1])):
						if j[1][w].upper() in self.special_words:
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
		with open("py_files/saved_objects/templates_processed_more_tuple.pickle","wb") as pickle_in:
			pickle.dump(temp_data,pickle_in)
		with open("py_files/saved_objects/templates_processed_more_tuple.pickle","rb") as pickle_in:
			self.templates= pickle.load(pickle_in)
		template_to_line=defaultdict(list)
		for i in ["second","third","fourth","fifth"]:
			for j in self.templates[i].keys():
				for k in self.templates[i][j]:
					template_to_line[" ".join(k[0])].append(k[1])
		with open("py_files/saved_objects/template_to_line.pickle","wb") as pickle_in:
			pickle.dump(template_to_line,pickle_in)
		with open("py_files/saved_objects/template_to_line.pickle","rb") as pickle_in:
			self.template_to_line=pickle.load(pickle_in)
		with open("py_files/saved_objects/pos_sylls_mode.p","rb") as pickle_in:
			self.pos_sylls_mode= pickle.load(pickle_in)
		with open("py_files/saved_objects/blacklist_index.p","rb") as pickle_in:
			self.blacklist_index= pickle.load(pickle_in)

		for i in self.special_words:
			try:
				self.pos_sylls_mode[i]=[(len(self.dict_meters[i.lower()][0]),1.0)]
			except:
				self.pos_sylls_mode[i]=[1,1.0]
	def helper(self,prompt):
		try:
			with open("py_files/saved_objects/spacy_prompt_to_w3s_rhyme_dict","rb") as pickle_in:
				mydict=pickle.load(pickle_in)
		except:
			with open("py_files/saved_objects/spacy_prompt_to_w3s_rhyme_dict","wb") as pickle_in:
				mydict={}
				pickle.dump(mydict,pickle_in)
		if prompt not in mydict.keys():
			w3s = self.get_similar_word_henry([prompt], n_return=20, word_set=set(self.filtered_nouns_verbs))
			w3s_rhyme_dict = {w3: {word for word in self.get_rhyming_words_one_step_henry(w3) if self.filter_common_word_henry(word, fast=True)} for w3 in w3s}
			mydict[prompt]=w3s_rhyme_dict
		self.w3s_rhyme_dict=mydict[prompt]
		with open("py_files/saved_objects/spacy_prompt_to_w3s_rhyme_dict","wb") as pickle_in:
			pickle.dump(mydict,pickle_in)
	'''
	def printing(self,data, f, f_final):
		with open(f_final+"_"+".pickle","rb") as pickle_in:
			data_old=pickle.load(pickle_in)
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
						line=list(self.template_to_line[" ".join(i_list)][0])+["\n"]
					except:
						line=list(self.template_to_line[" ".join(i_list[:-1])][0])+["\n"]
					lines+=line

			f.write("======================= template: {} ============================  \n".format(t+1))
			f.write(k)
			f.write("----------------------- original sentences ------------------------------------ \n")
			f.write(" ".join(lines))
			for j in temp_data[k]:
				adjusted_score=np.mean(j[1])+self.word_embedding_coefficient*np.mean(j[5])
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
		with open(f_final+"_"+".pickle","wb") as pickle_in:
			pickle.dump(data_curr,pickle_in)
	'''

	def gen_poem_andre_new(self, prompt, search_space, retain_space, word_embedding_coefficient=0,stress=True, prob_threshold=-10, mode="multi", relax_story_line=False,diversity=True):
		"""
		Generate poems with multiple templat es given a seed word (prompt) and GPT2
		search space.
        Parameters
        ----------
		prompt: str
			A seed word used to kickstart poetry generation.
        search_space : int
            Search space of the sentence finding algorithm.
            The larger the search space, the more sentences the network runs
            in parallel to find the best one with the highest score.
        retain_space : int
            How many sentences per template to keep.
		stress: bool
			Whether we enforce stress.
		prob_threshold: float
			If the probability of a word is lower than this threshold we will not consider
			this word. Set it to None to get rid of it.
		"""
		self.mode=mode
		self.relax_story_line=relax_story_line
		self.prob_threshold = prob_threshold
		self.enforce_stress = stress
		self.diversity=diversity
		if self.mode!="multi":
			self.relax_story_line=True
			self.prob_threshold = None
		self.finer_pos_category()
		self.create_w1s_rhyme_dict(prompt)
		print("=================== Finished Initializing ==================================")
		self.word_embedding_alpha = 0.5
		self.word_embedding_coefficient = word_embedding_coefficient
		self.n_w25_threshold=10
		print("===============================   helper       ==============================================")
		self.helper(prompt)
		print("===============================   end helper       ==============================================")
		self.madlib_verbs = self.get_madlib_verbs(prompt,["VBD", "VBN", "VB", "VBZ", "VBP", "VBG"])
		# get rid of common words
		if "was" in self.madlib_verbs["VBD"]:
			self.madlib_verbs["VBD"].remove("was")
			print("remove was \n")
		# self.madlib_verbs = self.get_madlib_verbs(prompt,["NN","NNS"])
		print("------- Madlib Verbs ------")
		print(self.madlib_verbs)
		#self.w1s_rhyme_dict= self.get_two_sets_20191113_henry(prompt,self.n_w25_threshold)
		#self.w3s_rhyme_dict=w3s_rhyme_dict
		
		#female_name_list, male_name_list=self.load_name_list()

		'''
		for name in self.w1s_rhyme_dict.keys():
			if name.lower() not in female_name_list and  name.lower() not in male_name_list:
				del self.w1s_rhyme_dict[name]
		'''
		print("=========================== Creating Wema =======================================")
		self.get_wema_dict_mp(prompt)
		print("=========================== Finished Wema =======================================")

		last_word_dict=self.last_word_dict(self.w1s_rhyme_dict,self.w3s_rhyme_dict)
		#self.saved_directory="limericks_experiment_TB_CLS"
		'''
		if self.saved_directory not in os.listdir(os.getcwd()):
			os.mkdir(self.saved_directory)
		if self.mode=="multi":
			result_file_path = self.saved_directory +"/"+ prompt+"_" + str(search_space)+"_"+str(retain_space)+"_"+str(self.word_embedding_coefficient)+"_"+self.mode+"_"+str(diversity)+"_"+"original"
		else:
			result_file_path = self.saved_directory +"/"+ prompt+"_" + str(search_space)+"_"+str(retain_space)+"_"+str(self.word_embedding_coefficient)+"_"+str(self.mode)+"_"+str(diversity)+"_"+"original"
		f = open(result_file_path+".txt","a+")
		'''
		previous_data=[]
		if self.get_spacy_similarity("Robert", prompt)>=self.get_spacy_similarity("Sarah",prompt):
			temp_name="Robert"
		else:
			temp_name="Sarah"
		candidates=self.gen_first_line_new(temp_name.lower(),search_space=5,strict=True,seed=prompt)
		assert len(candidates)>0, "no first line"
		print(candidates)
		for text in candidates:
			first_line_encodes = self.enc.encode(" ".join(text))
			previous_data.append((tuple(first_line_encodes),(0,),tuple(text)+("\n",), (text[-1],"\n"),("",""),(0,)))
		for which_line, num_sylls in zip(["second","third","fourth","fifth"],[9,6,6,9]):

			print("======================= starting {} line generation =============================".format(which_line))
			last_word_set=last_word_dict[which_line]
			possible=self.get_all_templates(num_sylls,which_line,last_word_set)
			previous_data=self.gen_line_flexible(previous_data=previous_data, possible=possible,num_sylls=num_sylls, search_space=search_space,retain_space=retain_space, which_line=which_line)

		return previous_data, self.template_to_line, self.words_to_names_rhyme_dict
		#self.printing(previous_data,f, f_final)

	def encodes_align(self,previous_data):
		"""
		Different lines have different encodes length. We force the encodes to
		have the same length as the minimal length encodes so that the encodes
		is a matrix that GPT2 accepts.
		"""
		encodes_length=[len(i[0]) for i in previous_data]
		encodes=[i[0][-min(encodes_length):] for i in previous_data]
		temp=[]
		for i,j in enumerate(previous_data):
			temp.append((encodes[i],j[1],j[2],j[3],j[4],j[5]))
		return temp

	def last_word_dict(self, w1s_rhyme_dict, w3s_rhyme_dict):
		"""
		Given the rhyme sets, extract all possible last words from the rhyme set
		dictionaries.

        Parameters
        ----------
		w1s_rhyme_dict: dictionary
			Format is {w1: [w2/w5s]}
		w3s_rhyme_dict: dictionary
			Format is {w3: [w4s]}

        Returns
        -------
        dictionary
            Format is {'second': ['apple', 'orange'], 'third': ['apple', orange] ... }

		"""
		last_word_dict={}
		for i in ["second","third","fourth","fifth"]:
			temp=[]
			if i== "fifth":
				for k in w1s_rhyme_dict.keys():
					temp+=w1s_rhyme_dict[k]
			if i=="second":
				for k in w1s_rhyme_dict.keys():
					temp.append(k)
			if i== "fourth":
				for k in w3s_rhyme_dict.keys():
					temp+=w3s_rhyme_dict[k]
			if i=="third":
				for k in w3s_rhyme_dict.keys():
					temp.append(k)
			last_word_dict[i]=[*{*temp}]
		return last_word_dict

	def sylls_bounds(self,partial_template):
		"""
		Return upper and lower bounds of syllables in a POS template.
		"""

		threshold=0.1
		sylls_up=0
		sylls_lo=0
		if len(partial_template)==1:
			return 2,1
		else:
			for t in partial_template[:-1]:
				x=[j[0] for j in self.pos_sylls_mode[t] if j[1]>=min(threshold,self.pos_sylls_mode[t][0][1])]
				if len(x)==0:
					sylls_up+=0
					sylls_lo+=0
				else:
					sylls_up+=max(x)
					sylls_lo+=min(x)
		sylls_up+=2
		sylls_lo+=1
		return sylls_up, sylls_lo

	def there_is_template_new(self,last_word_info,num_sylls, which_line):
		"""
		Return a list of possible templates given last word's POS, last word's syllabes
		and which line it is.
		"""
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

	def get_all_templates(self, num_sylls, which_line, last_word_set):
		"""
		Given number of syllables a line has, which line it is and all possible last
		words, return all possible POS templates
		"""

		last_word_info_set=set()
		temp=[]
		for i in last_word_set:
			if i in self.words_to_pos.keys() and i in self.dict_meters.keys():
				for j in self.get_word_pos(i):
					last_word_info_set.add((j,len(self.dict_meters[i][0])))
		for i in last_word_info_set:
			temp+=self.there_is_template_new(i, num_sylls, which_line)
		temp=[x for x in set(tuple(x) for x in temp)]
		return temp

	def template_sylls_checking(self, pos_set, sylls_set, template_curr, num_sylls_curr, possible, num_sylls):
		"""
		Check whether the current word could fit into our template with given syllables constraint

        Parameters
        ----------
		pos_set: set
			POS of the current word
		sylls_set: set
			Possible number of syllabes of the current word
		template_curr: list
			Partial, unfinished POS template of the current line (e.g. [NN, VB, NN])
		num_sylls_curr: int
			Syllable count of the partially constructed sentence
		possible: list
			All possible POS templates associated with the current line
		num_sylls: int
			predefined number of syllables the current line should have (e.g. 6,9)

        Returns
        -------
        list
            Format is [(POS, sylls)], a combination of possible POS
			and number of syllables of the current word
		"""
		continue_flag=set()
		for t in possible:
			if t[:len(template_curr)]==template_curr and len(t)>len(template_curr)+1:
				for pos in pos_set:
					if pos==t[len(template_curr)]:
						for sylls in sylls_set:
							sylls_up, sylls_lo=self.sylls_bounds(t[len(template_curr)+1:])
							if num_sylls-num_sylls_curr-sylls>=sylls_lo and num_sylls-num_sylls_curr-sylls<=sylls_up:
								continue_flag.add((pos,sylls))
		if len(continue_flag)==0: continue_flag=False
		return continue_flag

	def end_template_checking(self, pos_set, sylls_set, template_curr, num_sylls_curr, possible, num_sylls):
		"""
		Check whether the current word could fit into a template as the last word
		of the line with given syllables constraint

        Parameters
        ----------
		pos_set: set
			POS of the current word
		sylls_set: set
			Possible number of syllabes of the current word
		template_curr: list
			Partial, unfinished POS template of the current line (e.g. [NN, VB, NN])
		num_sylls_curr: int
			Syllable count of the partially constructed sentence
		possible: list
			All possible POS templates associated with the current line
		num_sylls: int
			predefined number of syllables the current line should have (e.g. 6,9)

        Returns
        -------
        list
            Format is [(POS, sylls)], a combination of possible POS
			and number of syllables of the current word
		"""

		end_flag=set()
		if self.mode=="multi" and self.relax_story_line==False:
			for t in possible:
				if t[:len(template_curr)]==template_curr and len(t)==len(template_curr)+1:
					for pos in pos_set:
						if pos==t[len(template_curr)]:
							for sylls in sylls_set:
								if num_sylls_curr+sylls==num_sylls:
									end_flag.add((pos,sylls))

		# this version, does not check last word pos.
		else:
			for t in possible:
				if t[:len(template_curr)]==template_curr and len(t)==len(template_curr)+1:
					for sylls in sylls_set:
						if num_sylls_curr+sylls==num_sylls:
							pos=t[len(template_curr)]
							end_flag.add((pos,sylls))
		if len(end_flag)==0:
			end_flag=False
		return end_flag


	def diversity_sort(self,search_space, retain_space,data, finished):
		"""
		Given a list of sentences, put them in bins according to their templates, get
		retain_space sentences from each bin and form a list, and get top search_space sentences from
		the list.

        Parameters
        ----------
		search_space: int
			Number of sentences returned
		data: list
			Input sentences
		finished: bool
			Whether the current sentence is completed
		"""
		if self.diversity:
			temp_data=defaultdict(set)
			# Key is "template; current_line_template". For each key we only keep retain_space sentences
			for n in data:
				if not finished:
					key=n[3]+n[4]
				else:
					key=n[3] # because the curr is already merged.
				temp_data[key].add(n)
			data=[]
			list_of_keys=list(temp_data.keys())
			x=random.sample(list_of_keys, len(list_of_keys))
			for k in x:
				if not finished:
					temp=heapq.nlargest(min(len(temp_data[k]),retain_space), temp_data[k], key=lambda x: np.mean(x[1]) + self.word_embedding_coefficient * x[7][-1])
					data.append((temp,np.max([np.mean(m[1])+self.word_embedding_coefficient * m[7][-1] for m in temp])))
				else:
					temp=heapq.nlargest(min(len(temp_data[k]),retain_space), temp_data[k], key=lambda x: np.mean(x[1]) + self.word_embedding_coefficient * x[5][-1])
					data.append((temp,np.max([np.mean(m[1])+self.word_embedding_coefficient * m[5][-1] for m in temp])))
			data=heapq.nlargest(min(len(data),search_space),data, key = lambda x: x[1])
			data_new=[]
			for k in data:
				data_new+=k[0]
			data=data_new
		else:
			if not finished:
				data_new=heapq.nlargest(min(len(data),search_space*retain_space), data, key=lambda x: np.mean(x[1]) + self.word_embedding_coefficient * x[7][-1])
			else:
				data_new=heapq.nlargest(min(len(data),search_space*retain_space), data, key=lambda x: np.mean(x[1]) + self.word_embedding_coefficient * x[5][-1])
			return data_new, 0

		return data_new, len(temp_data.keys())

	def get_word_pos(self, word):
		"""
		Get the set of POS category of a word. If we are unable to get the category, return None.
		"""
		if word not in self.words_to_pos:
			return None
		# Special case
		if word.upper() in self.special_words:
			return set([word.upper()])
		return set(self.words_to_pos[word])

	def get_wema_dict_mp(self,prompt):
		try:
			with open("py_files/saved_objects/poetic_wema_dict_{}_{}.pickle".format(prompt,self.n_w25_threshold),"rb") as pickle_in:
				self.wema_dict=pickle.load(pickle_in)
		except:
			with open("py_files/saved_objects/poetic_wema_dict_{}_{}.pickle".format(prompt,self.n_w25_threshold),"wb") as pickle_in:
				num_list_list= self.split_chunks(list(range(50256)))
				manager_wema = mp.Manager()
				output_wema=manager_wema.Queue()
				processes = [mp.Process(target=self.get_wema_dict, args=(num_list_list[i], output_wema)) for i in range(len(num_list_list))]
				print("******************************* create new wema, multiprocessing starts with {} processes *************************************".format(len(processes)))
				for p in processes:
					p.start()
				for p in processes:
					p.join()
				results = [output_wema.get() for p in processes]
				self.wema_dict=collections.defaultdict(dict)
				for r in results:
					for word in r.keys():
						self.wema_dict[word]=r[word]
				print(self.wema_dict["happy"])
				print("********************************** multiprocessing ends for wema *****************************************************")
				pickle.dump(self.wema_dict,pickle_in)
	# wema dict structure
	# {"happy" -> {"second": {"bee": 0.57 (avg distance to words that rhyme with bee), "cow": 0.69...}}}
	def get_wema_dict(self, num_list, output_wema):
		wema_dict=collections.defaultdict(dict)
		for index in num_list:
			# Decode index into word
			word = self.enc.decode([index]).lower().strip()
			# note that both , and . are in these keys()
			if len(word)==0 or word not in self.words_to_pos.keys() or word not in self.dict_meters.keys():
				continue

			pos_set=self.get_word_pos(word)
			sylls_set=set([len(m) for m in self.dict_meters[word]])
			if len(pos_set)==0 or len(sylls_set)==0:
				continue

			line_dict=collections.defaultdict(dict)
			# For each line, create a dictionary where the key is the rhyming word of the line
			# and the value is the average distance to the rhyme set corresponding to the rhyme word.
			for which_line in ["second","fifth", "third", "fourth"]:
				word_dict = collections.defaultdict()

				if which_line=="second":
					rhyme_dict = {"second_line_special_case": self.w1s_rhyme_dict.keys()}
				elif which_line=="fifth":
					rhyme_dict=self.w1s_rhyme_dict
				# Hack: for third line, we do not have a rhyming word yet, and the legitimate
				# ending word of the third line is chosen from the keys of our rhyming dictionary.
				elif which_line == "third":
					rhyme_dict = {"third_line_special_case": self.w3s_rhyme_dict.keys()}
				else:
					rhyme_dict = self.w3s_rhyme_dict

				for rhyme_word in rhyme_dict.keys():
					rhyme_set = rhyme_dict[rhyme_word]
					distance = self.get_word_similarity(word, rhyme_set)
					if distance is not None:
						word_dict[rhyme_word] = distance

				line_dict[which_line]=word_dict
			wema_dict[word]=line_dict
		output_wema.put(wema_dict)

	def get_word_embedding_moving_average(self, original_average, word, rhyme_word, which_line):
		"""
		Calculate word embedding moving average with the story line set selection.
		"""
		if which_line == "second" or which_line == "fifth":
			which_line = "second or fifth"
		if rhyme_word in self.wema_dict[word][which_line].keys():
			embedding_distance=self.wema_dict[word][which_line][rhyme_word]
			#return embedding_distance
		else:
			return original_average
		return (1 - self.word_embedding_alpha) * original_average + self.word_embedding_alpha * embedding_distance

	def split_chunks(self, data):
		data_list=[]
		chuck_len = len(data)//self.cpu + 1
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


	def gen_line_flexible(self, previous_data, possible, num_sylls, search_space, retain_space,which_line):
		'''
		Generate a line using multiple templates.

        Parameters
        ----------
		previous_data: list of tuples
			Each element has a tuple structure (encodes, score, text, template, (w1,w3)).
			encodes: list of int
				encodes are gpt-index for words
			score: double
				 Score is the probability of the line
			text: list of str
				the text corresponding to encodes
			template: list of POS
				template is all existing templates, e.g. if we are genrating third line right now, template is ["somename","second line templates"].
			(w1,w3):  tuple,
				It records the rhyme word in this sense, second line and fifth line last word have to be
				in the w1s_rhyme_dict[w1], the fourth line last word have to be in w3s_rhyme_dict[w3]. Note if we are only at line2,
				then w3 is '', because it hasn't happened yet.
		possible: list
			Possible templates for current line.
		search_space: int
			We generate search_space lines and sort them by probability to find out the bes line.
		num_sylls: int
			Number of syllables of current line
		which_line: int
			which line it is (1,2,3,4 or 5)
		'''
		previous_data=self.encodes_align(previous_data)
		sentences=[]
		for i in previous_data:
			template_curr=()
			num_sylls_curr=0
			sentences.append([i[0],i[1],i[2],i[3],template_curr,num_sylls_curr,i[4], i[5]])
		# sentences is a tuple, each element looks like (encodes, score, text, template, current_line_template, how_many_syllabus_used_in_current_line, (w1,w3), moving average/word similarity list)
		# curren_line_template is a partial template of the currently developing line. template is all the POS of the developing poem, with lines separated by "\n".
		finished_sentences=[]
		iteration=0
		new_sentences=[1]
		while(len(new_sentences)>0):
			iteration+=1
			context_token=[s[0] for s in sentences]
			m=len(context_token)
			context_token=np.array(context_token).reshape(m,-1)
			print("******************************** gpt2 Starts Processing Next Word **********************************")
			logits = score_model(model_name=self.model_name, context_token = context_token)
			print("******************************** gpt2 Finished Processing Next Word **********************************")
			'''
			logits_list= self.split_chunks(logits)
			sentences_list=self.split_chunks(sentences)
			manager = mp.Manager()
			output=manager.Queue()
			processes = [mp.Process(target=self.batch_process_word, args=(which_line, possible,num_sylls,logits_list[mp_index], sentences_list[mp_index], output)) for mp_index in range(len(logits_list)) ]
			print("******************************** multiprocessing starts with {} processes *************************************".format(len(processes)))
			for p in processes:
				p.start()

			for p in processes:
				p.join()
			print("********************************** multiprocessing ends *****************************************************")
			results = [output.get() for p in processes]
			new_sentences, quasi_finished_sentences = [], []
			for result in results:
				new_sentences += result[0]
				quasi_finished_sentences += result[1]
			'''
			new_sentences, quasi_finished_sentences= self.batch_process_word(which_line, possible, num_sylls, logits, sentences)
			if self.punctuation[which_line]:
				if len(quasi_finished_sentences)>0:
					quasi_finished_sentences, diversity=self.diversity_sort(search_space,retain_space,quasi_finished_sentences, finished=True)
					context_token=[s[0] for s in quasi_finished_sentences]
					m=len(context_token)
					context_token=np.array(context_token).reshape(m,-1)
					print("################################## gpt2 Starts Adding Punctuation #############################")
					logits = score_model(model_name=self.model_name, context_token = context_token)
					print("################################## gpt2 Finished Adding Punctuation #############################")
					for i,j in enumerate(logits):
						sorted_index=np.argsort(-1*j)
						for index in sorted_index:
							word = self.enc.decode([index]).lower().strip()
							if word==self.sentence_to_punctuation[which_line]:
								finished_sentences.append((quasi_finished_sentences[i][0] + (index,),
															quasi_finished_sentences[i][1] + (np.log(j[index]),),
															quasi_finished_sentences[i][2]+(word,),
															quasi_finished_sentences[i][3]+(word,),
															quasi_finished_sentences[i][4],
															quasi_finished_sentences[i][5]))
								break
			else:
				for q in quasi_finished_sentences:
					finished_sentences.append(q)
			print("\n ========================= iteration {} ends =============================".format(iteration))
			sentences, diversity=self.diversity_sort(search_space,retain_space,new_sentences, finished=False)
			print("{} sentences before diversity_sort, {} sentences afterwards, diversity {}, this iteration has {} quasi_finished_sentences,  now {} finished_sentences \n".format(len(new_sentences),len(sentences), diversity, len(quasi_finished_sentences),len(finished_sentences)))
		assert len(sentences)==0, "something wrong"
		previous_data_temp, _=self.diversity_sort(search_space,retain_space,finished_sentences, finished=True)
		previous_data=[(i[0],i[1],i[2]+("\n",),i[3]+("\n",),i[4],i[5]+(0,)) for i in previous_data_temp]
		return previous_data

	def get_madlib_verbs(self, prompt, pos_list, n_return=20):
		# dictionary {pos: set()}
		try:
			pickle_in=open("py_files/saved_objects/spacy_prompt_to_madlib_verbs.pickle","rb")
			mydict=pickle.load(pickle_in)
			pickle_in.close()
		except:
			pickle_in=open("py_files/saved_objects/spacy_prompt_to_madlib_verbs.pickle","wb")
			mydict={}
			pickle.dump(mydict,pickle_in)
			pickle_in.close()
		if prompt in mydict.keys() and mydict[prompt]!=None:
			return mydict[prompt]
		else:
			ret={pos: self.get_similar_word_henry([prompt], n_return=n_return, word_set=set(self.pos_to_words[pos]))for pos in pos_list}
			mydict[prompt]=ret
			with open("py_files/saved_objects/spacy_prompt_to_madlib_verbs.pickle","wb") as pickle_in:
				pickle.dump(mydict,pickle_in)
			return ret

	def batch_process_word(self, which_line, possible, num_sylls, logits, sentences, output=None, madlib_flag=True):
		'''
		Batch process the new possible word of a group of incomplete sentences.

        Parameters
        ----------
		possible: list
			list of possible templates
		num_sylls: int
			we generate search_space lines and sort them by probability to find out the bes line.
		which_line: int
			which line it is (1,2,3,4 or 5)
		num_sylls: int
			wumber of syllables of current line.
		logits: list
			Logits is the output of GPT model.
		sentences: list
			List of sentences that we currently are generating.
		'''

		new_sentences = []
		quasi_finished_sentences = []
		for i,j in enumerate(logits):
			sorted_index=np.argsort(-1*j)
			word_list_against_duplication=[]
			# sentences is a tuple, each element looks like (encodes, score, text, template, current_line_template, how_many_syllabus_used_in_current_line, (w1,w3), moving average)
			# curren_line_template is a partial template of the currently developing line.
			#template is all the POS of the developing poem, with lines separated by "\n".
			template_curr=sentences[i][4]
			num_sylls_curr=sentences[i][5]
			moving_avg_curr=sentences[i][7][-1]
			rhyme_set_curr = set()
			if which_line=="second":
				rhyme_set_curr = self.w1s_rhyme_dict.keys()
				rhyme_word="second_line_special_case"
			if which_line=="fifth":
				rhyme_set_curr=self.w1s_rhyme_dict[sentences[i][6][0]]
				rhyme_word=sentences[i][6][0]
			if which_line=="third":
				rhyme_set_curr = self.w3s_rhyme_dict.keys()
				rhyme_word="third_line_special_case"
			if which_line=="fourth":
				rhyme_set_curr = self.w3s_rhyme_dict[sentences[i][6][1]]
				rhyme_word=sentences[i][6][1]
			pdb.set_trace()

			# If it is the fifth line, the current template has to corresponds to the fourth line template
			# because they are usually one sentence

			for ii,index in enumerate(sorted_index):
				# Get current line's template, word embedding average, word, rhyme set, etc.
				word = self.enc.decode([index]).lower().strip()
				if self.prob_threshold is not None and np.log(j[index]) < self.prob_threshold:
					break
				if word in word_list_against_duplication:
					continue
				elif len(word)==0:
					continue
				# note that both , and . are in these keys()
				elif word not in self.words_to_pos.keys() or word not in self.dict_meters.keys():
					continue
				else:
					if index in self.blacklist_index:
						continue
					pos_set=self.get_word_pos(word)
					sylls_set=set([len(m) for m in self.dict_meters[word]])
					if len(pos_set)==0 or len(sylls_set)==0:
						continue
					# If the word is a noun or adjective and has appeared
					# previously, we discard the sentence.
					if self.is_duplicate_in_previous_words(word, sentences[i][2]):
						continue

					# If stress is incorrect, continue
					if self.enforce_stress:
						possible_syllables = self.dict_meters[word]
						word_length = min(sylls_set)

						stress = [1, 4] if (which_line == "third" or which_line == "fourth") else [1, 4, 7]
						correct_stress = True
						# There is a stress on current word
						for stress_position in stress:
							if num_sylls_curr <= stress_position and num_sylls_curr + word_length > stress_position:
								stress_syllable_pos = stress_position - num_sylls_curr
								if all(s[stress_syllable_pos] != '1' for s in possible_syllables):
									correct_stress = False
								break
						if not correct_stress:
							continue

					# end_flag is the (POS, Sylls) of word if word can be the last_word for a template, False if not
					# continue_flag is (POS,Sylls) if word can be in a template and is not the last word. False if not
					continue_flag=self.template_sylls_checking(pos_set=pos_set,sylls_set=sylls_set,template_curr=template_curr,num_sylls_curr=num_sylls_curr,possible=possible, num_sylls=num_sylls)
					end_flag=self.end_template_checking(pos_set=pos_set,sylls_set=sylls_set,template_curr=template_curr,num_sylls_curr=num_sylls_curr,possible=possible, num_sylls=num_sylls)
					word_embedding_moving_average = self.get_word_embedding_moving_average(moving_avg_curr, word, rhyme_word, which_line)
					tuple_of_wema=tuple([m for m in sentences[i][7][:-1]])+(word_embedding_moving_average,)
					if continue_flag:
						word_list_against_duplication.append(word)
						for continue_sub_flag in continue_flag:

							# If current word POS is VB, current line is second line and word is not in our
							# precomputed list, throw away the sentence
							if madlib_flag:
								curr_vb_pos = continue_sub_flag[0]
								if 'VB' in curr_vb_pos and which_line == 'second' \
									and not any('VB' in pos_tag for pos_tag in template_curr):
									if word not in self.madlib_verbs[curr_vb_pos]:
										continue
							word_tuple = (sentences[i][0] + (index,),
												sentences[i][1] + (np.log(j[index]),),
												sentences[i][2]+(word,),
												sentences[i][3],
												sentences[i][4]+(continue_sub_flag[0],),
												sentences[i][5]+continue_sub_flag[1],
												sentences[i][6],
												tuple_of_wema)
							new_sentences.append(word_tuple)
					if end_flag:
						for end_sub_flag in end_flag:
							if which_line=="second":
								if word in rhyme_set_curr:
									word_list_against_duplication.append(word)
									word_tuple=(sentences[i][0] + (index,),
												sentences[i][1] + (np.log(j[index]),),
												sentences[i][2]+(word,),
												sentences[i][3]+sentences[i][4]+(end_sub_flag[0],),
												(word,""),
												tuple_of_wema)
									quasi_finished_sentences.append(word_tuple)
							if which_line=="third":
								if word in rhyme_set_curr:
									word_list_against_duplication.append(word)
									word_tuple=(sentences[i][0] + (index,),
												sentences[i][1] + (np.log(j[index]),),
												sentences[i][2]+(word,),
												sentences[i][3]+sentences[i][4]+(end_sub_flag[0],),
												(sentences[i][6][0],word),
												tuple_of_wema)
									quasi_finished_sentences.append(word_tuple)
							if which_line=="fourth" or which_line=="fifth":
								if word in rhyme_set_curr:
									word_list_against_duplication.append(word)
									word_tuple=(sentences[i][0] + (index,),
												sentences[i][1] + (np.log(j[index]),),
												sentences[i][2]+(word,),
												sentences[i][3]+sentences[i][4]+(end_sub_flag[0],),
												sentences[i][6],
												tuple_of_wema)
									quasi_finished_sentences.append(word_tuple)
		#output.put((new_sentences, quasi_finished_sentences))
		return new_sentences, quasi_finished_sentences
