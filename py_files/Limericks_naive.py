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
		self.female_name_list, self.male_name_list = pickle.load(open("py_files/saved_objects/name_list.p", "rb"))
		self.sentence_to_punctuation={"second":".","third":",","fourth":",","fifth":"."}
		with open(self.filtered_names_rhymes, "rb") as hf:
			self.names_rhymes_list = pickle.load(hf)
		# word embedding coefficients

	def different_gender(self,gender):
		temp=[]
		for i in self.names_rhymes_list:
			male=[]
			female=[]
			for j in i[0]:
				if j in self.male_name_list:
					male.append(j)
				if j in self.female_name_list:
					female.append(j)
			if gender=="male":
				if len(male)>0:
					temp.append((male,i[1]))
			if gender=="female":
				if len(female)>0:
					temp.append((female,i[1]))
		self.names_rhymes_list=temp

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


	def gen_poem_andre_new(self, gender,prompt, search_space, retain_space, word_embedding_coefficient=0,stress=True, prob_threshold=-10):
		self.story_line=False
		if gender=="male":
			temp_name="Robert"
		if gender=="female":
			temp_name="Sarah"
		self.different_gender(gender)
		self.create_w1s_rhyme_dict(prompt)
		if not self.story_line:
			try:
				pickle_in=open("py_files/saved_objects/rhyming_dict_for_no_storyline.pickle","rb")
				self.rhyming_dict=pickle.load(pickle_in)
				pickle_in.close()
			except:
				pickle_in=open("py_files/saved_objects/rhyming_dict_for_no_storyline.pickle","wb")
				self.rhyming_dict={}
				pickle.dump(self.rhyming_dict,pickle_in)
				pickle_in.close()
		self.prob_threshold = prob_threshold
		self.enforce_stress = stress
		print("=================== Finished Initializing ==================================")
		self.word_embedding_alpha = 0.5
		self.word_embedding_coefficient = word_embedding_coefficient
		
		previous_data=[]
		candidates=self.gen_first_line_new(temp_name.lower(),search_space=5,strict=True,seed=prompt)
		assert len(candidates)>0, "no first line"
		text=random.choice(candidates)
		first_line_encodes = self.enc.encode(" ".join(text))
		previous_data.append((tuple(first_line_encodes),(0,),tuple(text)+("\n",), (text[-1],"\n"),("",""),(0,)))



		for which_line, num_sylls in zip(["second","third","fourth","fifth"],[9,6,6,9]):
		#for which_line, num_sylls in zip(["third"],[6]):

			print("======================= starting {} line generation =============================".format(which_line))
			previous_data=self.gen_line_flexible(previous_data=previous_data, num_sylls=num_sylls, search_space=search_space,retain_space=retain_space, which_line=which_line)
		with open("py_files/saved_objects/rhyming_dict_for_no_storyline.pickle","wb") as pickle_in:
			pickle.dump(self.rhyming_dict,pickle_in)
		# Print out generated poems
		#self.printing(previous_data,f, f_final, counter)
		previous_data, _ = self.diversity_sort(data=previous_data,last=True)
		return previous_data, self.template_to_line, self.words_to_names_rhyme_dict
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


	def template_sylls_checking(self, sylls_set, template_curr, num_sylls_curr, num_sylls):
		continue_flag=set()
		for sylls in sylls_set:
			if num_sylls-num_sylls_curr-sylls>=1:
				continue_flag.add(("POS_NULL",sylls))
		if len(continue_flag)==0: continue_flag=False
		return continue_flag

	def end_template_checking(self, sylls_set, template_curr, num_sylls_curr, num_sylls):
		end_flag=set()
		for sylls in sylls_set:
			if num_sylls_curr+sylls==num_sylls:
				end_flag.add((pos,sylls))
		if len(end_flag)==0:
			end_flag=False
		return end_flag


	def diversity_sort(self,search_space=None, retain_space=None,data=None, finished=None,last=False):
		if last:
			data_new=heapq.nlargest(len(data), data, key=lambda x: np.mean(x[1]))
			return data_new,0
		
		if not finished:
			data_new=heapq.nlargest(min(len(data),search_space*retain_space), data, key=lambda x: np.mean(x[1]) + self.word_embedding_coefficient * x[7][-1])
		else:
			data_new=heapq.nlargest(min(len(data),search_space*retain_space), data, key=lambda x: np.mean(x[1]) + self.word_embedding_coefficient * x[5][-1])
		return data_new, 0

		return data_new, len(temp_data.keys())

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


	def gen_line_flexible(self, previous_data, num_sylls, search_space, retain_space,which_line):
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
			rhyme_word_set=set()
			for sent in sentences:
				if which_line=="fifth":rhyme_word_set.add(sent[6][0])
				if which_line=="fourth":rhyme_word_set.add(sent[6][1])
			for rhyme_word in rhyme_word_set:
				if rhyme_word not in self.rhyming_dict.keys():
					self.rhyming_dict[rhyme_word]=self.get_rhyming_words_one_step_henry(word=rhyme_word)
			
			logits_list= self.split_chunks(logits)
			sentences_list=self.split_chunks(sentences)
			manager = mp.Manager()
			output=manager.Queue()
			processes = [mp.Process(target=self.batch_process_word, args=(which_line,num_sylls,logits_list[mp_index], sentences_list[mp_index], output)) for mp_index in range(len(logits_list)) ]
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
			
			#new_sentences, quasi_finished_sentences= self.batch_process_word(which_line, possible, num_sylls, logits, sentences)
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

	def batch_process_word(self, which_line,num_sylls, logits, sentences, output=None):
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
			rhyme_set_curr = set()
			if which_line=="fifth":
				rhyme_word=sentences[i][6][0]
				rhyme_set_curr=self.rhyming_dict[rhyme_word]
			if which_line=="fourth":
				rhyme_word=sentences[i][6][1]
				rhyme_set_curr=self.rhyming_dict[rhyme_word]

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
				elif  word not in self.dict_meters.keys():
					continue
				else:
					if index in self.blacklist_index:
						continue
					sylls_set=set([len(m) for m in self.dict_meters[word]])
					if len(sylls_set)==0:
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
					continue_flag=self.template_sylls_checking(sylls_set=sylls_set,template_curr=template_curr,num_sylls_curr=num_sylls_curr, num_sylls=num_sylls)
					end_flag=self.end_template_checking(sylls_set=sylls_set,template_curr=template_curr,num_sylls_curr=num_sylls_curr, num_sylls=num_sylls)
					if continue_flag:
						word_list_against_duplication.append(word)
						for continue_sub_flag in continue_flag:
							word_tuple = (sentences[i][0] + (index,),
												sentences[i][1] + (np.log(j[index]),),
												sentences[i][2]+(word,),
												sentences[i][3],
												sentences[i][4]+(continue_sub_flag[0],),
												sentences[i][5]+continue_sub_flag[1],
												sentences[i][6],
												sentences[i][7])
							new_sentences.append(word_tuple)
					if end_flag:
						for end_sub_flag in end_flag:
							if which_line=="second":
								if word in self.storyline_second_words:
									word_list_against_duplication.append(word)
									word_tuple=(sentences[i][0] + (index,),
												sentences[i][1] + (np.log(j[index]),),
												sentences[i][2]+(word,),
												sentences[i][3]+sentences[i][4]+(end_sub_flag[0],),
												(word,""),
												sentences[i][7])
									quasi_finished_sentences.append(word_tuple)
							if which_line=="third":
								word_list_against_duplication.append(word)
								word_tuple=(sentences[i][0] + (index,),
											sentences[i][1] + (np.log(j[index]),),
											sentences[i][2]+(word,),
											sentences[i][3]+sentences[i][4]+(end_sub_flag[0],),
											(sentences[i][6][0],word),
											sentences[i][7])
								quasi_finished_sentences.append(word_tuple)
							if which_line=="fourth":
								if word in rhyme_set_curr:
									word_list_against_duplication.append(word)
									word_tuple=(sentences[i][0] + (index,),
												sentences[i][1] + (np.log(j[index]),),
												sentences[i][2]+(word,),
												sentences[i][3]+sentences[i][4]+(end_sub_flag[0],),
												sentences[i][6],
												sentences[i][7])
									quasi_finished_sentences.append(word_tuple)
							if which_line=="fifth":
								if word in rhyme_set_curr:
									word_list_against_duplication.append(word)
									word_tuple=(sentences[i][0] + (index,),
												sentences[i][1] + (np.log(j[index]),),
												sentences[i][2]+(word,),
												sentences[i][3]+sentences[i][4]+(end_sub_flag[0],),
												sentences[i][6],
												sentences[i][7])
									quasi_finished_sentences.append(word_tuple)
		output.put((new_sentences, quasi_finished_sentences))
		#return new_sentences, quasi_finished_sentences
