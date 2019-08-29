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
	'''
	def configure_story_line(self,prompt):
		
		story_line=self.get_five_words_henry(prompt)
		valid_story_line=[]
		female_name_list, male_name_list=self.load_name_list()
		for i in story_line:
			if i[0].lower() in female_name_list or i[0].lower() in male_name_list:
				temp=0
				for (a,b,c) in zip([9,6,[6,9]],[i[1].lower(),i[2].lower(),[i[3].lower(),i[4].lower()]],["second","third","last2"]):
					templates=self.there_is_template(a,b,c)
					if templates:
						temp+=1
				if temp==3:
					valid_story_line.append(i)
		valid_story_line_pos=defaultdict(list)
		for v in valid_story_line:
			pos=[self.words_to_pos[w] for w in v]
			keys=["-".join([a,b,c,d]) for a in pos[1] for b in pos[2] for c in pos[3] for d in pos[4]]
			for k in keys:
				valid_story_line_pos[k].append(v)
		# I am assuming that every words in the same position have the same num of syllabus
		best_pos=sorted(valid_story_line_pos.keys(),key=lambda k:len(valid_story_line_pos[k]),reverse=True)
		five_words_cluster=valid_story_line_pos[best_pos[0]]
		temp=[]
		for i in range(5):
			temp1=[]
			for j in five_words_cluster:	
				temp1.append(j[i])
			temp.append(list(set(temp1)))
		five_words_cluster=temp
		for i in range(4):
			temp=self.there_is_template()
		'''
	def there_is_template(self,num_sylls,last_word,which_line):
	    """
	    Check whether there is a good template for the given last word and the given line and the given syllabus constraint

	    Parameters
	    ----------
	    num_sylls : integer or list of integers
	        when which_line="last2", num_sylls is a list of 2 integers, respectively the syllabus constraint for fourth and the fifth lines
	    last_word : str or list of str
	        when which_line="last2", last_word is a list of 2 str, the last words respectively for the last 2 lines
	    which_line: str
	        it can take "second","third","last2"

	    Returns
	    -------
	    it either returns a Boolean "False" or possible, which is a list of possible templates for the given line
	    note, when which_line=="last2", the element in the returned list is each a list of 2 templates for the last 2 lines, (template4, template5, line4, line5)
	    """
	    pickle_in = open("py_files/saved_objects/templates_new2.pickle","rb")
	    templates= pickle.load(pickle_in)
	    pickle_in.close()
	    dataset=templates[which_line]
	    if which_line=="second" or which_line=="third":
	    	last_word=last_word.lower().strip()
	    	x=set(self.words_to_pos[last_word])
	    	y=set(dataset.keys())
	    	z=x.intersection(y)
	    	z=list(z)
	        ### POS chec
	    	if len(z)==0:
	        	return False 
	        ### Meters check
	    	possible=[]
	    	for i in z:
	        	for j in dataset[i]:
	        		if j[2][0]<j[2][1]:
	        			if num_sylls>=j[2][0] and num_sylls<=j[2][1] and self.valid_permutation_sylls(num_sylls, j[0],len(self.dict_meters[last_word][0]))!=0:
	        				possible.append((j[0],j[1]))
	        		else:
	        			if num_sylls>=j[2][0] and num_sylls<=j[2][1]+1 and self.valid_permutation_sylls(num_sylls, j[0],len(self.dict_meters[last_word][0]))!=0:
	        				possible.append((j[0],j[1]))
	    	if len(possible)==0:
	        	return False
	    if which_line=="last2":
	    	last_word=[l.lower().strip() for l in last_word]
	    	pos_4=self.words_to_pos[last_word[0]]
	    	pos_5=self.words_to_pos[last_word[1]]
	    	x=set(['-'.join((m,n)) for m in pos_4 for n in pos_5])
	    	y=set(dataset.keys())
	    	z=x.intersection(y)
	    	z=list(z)
	        ### POS check
	    	if len(z)==0:
	        	return False
	        ### Meters check
	    	possible=[]
	    	for i in z:
	        	for j in dataset[i]:
	        		flag=[0,0]
	        		if j[4][0]<j[4][1]:
	        			if num_sylls[0]>=j[4][0] and \
	        			num_sylls[0]<=j[4][1] and \
	        			self.valid_permutation_sylls(num_sylls[0], j[0],len(self.dict_meters[last_word[0]][0]))!=None:
	        				flag[0]=1
	        		if j[4][0]==j[4][1]:
	        			if num_sylls[0]>=j[4][0] and \
	        			num_sylls[0]<=j[4][1]+1 and \
	        			self.valid_permutation_sylls(num_sylls[0], j[0],len(self.dict_meters[last_word[0]][0]))!=None:
	        				flag[0]=1
	        		if j[5][0]<j[5][1]:
	        			if num_sylls[1]>=j[5][0] and num_sylls[1]<=j[5][1] and self.valid_permutation_sylls(num_sylls[1], j[1],len(self.dict_meters[last_word[1]][0]))!=None:
	        				flag[1]=1
	        		if j[5][0]==j[5][1]:
	        			if num_sylls[1]>=j[5][0] and num_sylls[1]<=j[5][1]+1 and self.valid_permutation_sylls(num_sylls[1], j[1],len(self.dict_meters[last_word[1]][0]))!=None:
	        				flag[1]=1
	        		if sum(flag)==2:
	        			possible.append((j[0],j[1],j[2],j[3]))
	    	if len(possible)==0:
	    		return False 
	    return possible
	def gen_poem_andre(self, prompt, search_space=100,thresh_hold=10):
		story_line=self.get_five_words_henry(prompt)
		valid_story_line=[]
		female_name_list, male_name_list=self.load_name_list()
		for i in story_line:
			if i[0].lower() in female_name_list or i[0].lower() in male_name_list:
				temp=0
				for (a,b,c) in zip([9,6,[6,9]],[i[1].lower(),i[2].lower(),[i[3].lower(),i[4].lower()]],["second","third","last2"]):
					templates=self.there_is_template(a,b,c)
					if templates:
						temp+=1
				if temp==3:
					valid_story_line.append(i)
		five_words=random.choice(valid_story_line)
		with open("limericks_data/"+"_".join(five_words)+"_"+str(search_space)+"_"+str(thresh_hold)+".txt","a+") as f:
			print("data_path")
			print("limericks_data/"+"_".join(five_words)+"_"+str(search_space)+"_"+str(thresh_hold)+".txt")
			f.write(" ".join(five_words)+"\n")
			first_line=random.choice(self.gen_first_line_new(five_words[0].lower(),strict=True))
			poem=[first_line+["\n"]]
			print(first_line)
			print(five_words)
			f.write(" ".join(first_line)+"\n")
			first_line_encodes = self.enc.encode(" ".join(first_line))
			prompt=[(first_line_encodes,0, first_line, ["first_line"])]


			print("================================================= second_line ==============================================")
			possible=self.there_is_template(num_sylls=9,last_word=five_words[1],which_line="second")
			assert len(possible)>0, "no template found"
			record_old_return=self.gen_line_gpt_andre_multi_template(encodes=prompt, num_sylls=9,templates=possible, rhyme_word=five_words[1], search_space=search_space, comment=True, thresh_hold=thresh_hold)
			new_poem=[]
			print("================================================= third_line ================================================")
			possible=self.there_is_template(num_sylls=6,last_word=five_words[2],which_line="third")
			assert len(possible)>0, "no template found"
			possible_prompts=[]
			print("how many keys:", len(record_old_return.keys()))
			for k in record_old_return.keys():
			    temp= heapq.nlargest(min(3,len(record_old_return[k])), record_old_return[k], key=lambda x: x[2])
			    for i in temp:
			        for j in range(len(prompt)):
			            new_poem.append(poem[j]+i[0]+["\n"])
			            possible_prompts.append((i[1],i[2],i[0],re.findall(r'\S+|\n',k)))
			prompt=possible_prompts
			print("len(prompt):",len(prompt))
			poem=new_poem
			print("len(poem):",len(poem))
			record_old_return=self.gen_line_gpt_andre_multi_template(encodes=prompt, num_sylls=6,templates=possible, rhyme_word=five_words[2], search_space=search_space,comment=True, thresh_hold=thresh_hold)
			new_poem=[]

			print('==================================================== fourth line =============================================')

			possible=self.there_is_template(num_sylls=[6,9],last_word=(five_words[3],five_words[4]),which_line="last2")
			assert len(possible)>0, "no template found"
			possible_prompts=[]
			print("how many keys:", len(record_old_return.keys()))
			for k in record_old_return.keys():
			    temp= heapq.nlargest(min(3,len(record_old_return[k])), record_old_return[k], key=lambda x: x[2])
			    for i in temp:
			        for j in range(len(prompt)):
			            possible_prompts.append((i[1],i[2],i[0],re.findall(r'\S+|\n',k)))
			            new_poem.append(poem[j]+i[0]+["\n"])
			prompt=possible_prompts
			print("len(prompt):",len(prompt))
			poem=new_poem
			print("len(poem):", len(poem))
			#print(poem)
			template_4=[]
			template_5=[]
			mydict={}
			mydict2={}
			for i in possible:
			    template_4.append(i[0])
			    template_5.append(i[1])
			    try:
			        mydict[' '.join(i[0])].append(i[1])
			    except:
			        mydict[' '.join(i[0])]=[i[1]]
			    try:
			        mydict2[' '.join(i[1])].append(" ".join(i[0]))
			    except:
			        mydict2[' '.join(i[1])]=[" ".join(i[0])]

			record_old_return=self.gen_line_gpt_andre_multi_template(encodes=prompt, num_sylls=6,templates=template_4, rhyme_word=five_words[3], search_space=search_space,comment=True, thresh_hold=thresh_hold)
			new_poem={}
			print("================================================== fifth line ====================================================")

			possible_prompts={}

			print("how many keys:", len(record_old_return.keys()))


			for k in record_old_return.keys():
			    temp= heapq.nlargest(min(3,len(record_old_return[k])), record_old_return[k], key=lambda x: x[2])
			    for i in temp:
			        for j in range(len(prompt)):
			            try:
			                new_poem[k].append(poem[j]+i[0]+["\n"])
			                possible_prompts[k].append((i[1],i[2],i[0],re.findall(r'\S+|\n',k)))
			            except:
			                new_poem[k]=[poem[j]+i[0]+["\n"]]
			                possible_prompts[k]=[(i[1],i[2],i[0],re.findall(r'\S+|\n',k))]
			prompt=possible_prompts
			print("len(prompt):",len(prompt))
			poem=new_poem
			print("len(poem):", len(poem))





			record_old_return=self.gen_line_gpt_andre_multi_template(encodes=prompt, num_sylls=9,templates=template_5, rhyme_word=five_words[4], search_space=search_space, comment=True, mydict=mydict, thresh_hold=thresh_hold)

			pickle_out = open("andre_prompt.pickle","wb")
			pickle.dump(record_old_return, pickle_out)
			pickle_out.close()

			new_poem={}
			for k in record_old_return.keys():
			    k_temp=k.split("\n")[-1].strip()
			    temp= heapq.nlargest(min(len(record_old_return[k]),5), record_old_return[k], key=lambda x: x[2])
			    for i in temp:
			        try:
			            new_poem[k].append((i[0],i[2],re.findall(r'\S+|\n',k)))
			        except:
			            new_poem[k]=[(i[0],i[2],re.findall(r'\S+|\n',k))]
			#print(new_poem)
			best=[]
			for t, k in enumerate(new_poem.keys()):
			    f.write("======================================== template"+"*"+str(t+1)+"========================================"+"\n")
			    i=new_poem[k][0]
			    f.write(" ".join(i[2])+"\n")
			    for j,i in enumerate(new_poem[k]):
			        if j==0:
			            best.append(i)
			        f.write("---------------------------------------#"+str(j+1)+":"+str(i[1])+"---------------------------------------"+"\n")
			        f.write(" ".join(i[0])+"\n")
			f.write("\n"+"================================== Best 5 Poems ===================================================="+"\n")
			temp=heapq.nlargest(min(len(best),5), best, key=lambda x: x[1])
			for j,i in enumerate(temp):
			    f.write("=========================== Number"+str(j+1)+"Score"+ str(i[1])+"============================================="+"\n")
			    f.write(" ".join(i[0])+"\n")


		print("data_path")
		print("limericks_data/"+"_".join(five_words)+"_"+str(search_space)+"_"+str(thresh_hold)+".txt")
	def gen_line_gpt_andre_multi_template(self,encodes, num_sylls,templates=None, rhyme_word=None, search_space=100,comment=True,mydict=False,verbose=False, thresh_hold=10):
	    """
	    Uses GPT to generate multiple lines using all feasible templates conditioned upon the number of syllabus constraint, the last word cosntraint, and the also conditioned
	    on the templates passed into it, the encodes passed into it. 

	    Parameters
	    ----------
	    encodes : dict or list of tuples
	        encodes is a list of tuples [(encodes,score，sentences, tem)], where encodes is gpt2-byte encodes numbers; score is log likelihood of the prompt (encodes) passed
	        into it; sentences are a list of words directly associtated with gpt2-byte encodes numbers. tem is the last template, for example, if the previous sentences is line4, then tem is the line4 template
	        encodes can also be a dict, whose values are the same list of tuples, but whose keys are " ".join(whole templates), for example, the templates of first line+"\n" +second line +"\n"+third line
	    
	    num_sylls : num
	        num_sylls is the number of sullabus permissible for this line, usually, line1,line2,line5 has syllabus count 9, line3, lin4 has syllabus count 6

	    templates : list
	        list of templates that satisfy the syllabus constraint and last word POS constraint, given usually by self.there_is_template\

	    rhyme_word : str
	        the rhyme_word is the last word of the sentences, all sentences produced by this function will have the rhyme_word as the last word, it's usually from the story_line
	    
	    search_space: num
	        how wide the beam of the beam search is, this is directly related to how much memory the GPU has, usually search_space is set to 100-200

	    comment: Bool
	        turning this to False would suppress all comments, default True
	    mydict: Bool or dict
	        unless the sentece is last line of the limericks, whhose templates we suspect to be closely related to the templated of the fourth line, mydict is False,
	        otherwise, mydict is a dictionary that maps each fourth line template, with all its possible 5line templates.
	    debug: Bool
	        for dev, you can use this if statement to do a pdb.set_trace(), to debug

	    Returns
	    -------
	    record_old_return: dict
	        a dict whose keys are combined_long templates, and values (sentences, encodes, scores,tem)
	    """

	    # for now the rhyme_word is a word, but soon, it can be a list of words. 
	   



	    
	    # check if the encodes have the same legnth, if not, first now we go with the legnth with the most elements
	    length_list=[]
	    if mydict:
	        for k in encodes.keys():
	            for i in encodes[k]:
	                length_list.append(len(i[0]))
	        temp=max(set(length_list), key=length_list.count)
	        encodes_temp={}
	        for k in encodes.keys():
	            for i in encodes[k]:
	                if len(i[0])==temp:
	                    try:
	                        encodes_temp[k].append(i)
	                    except:
	                        encodes_temp[k]=[i]
	    else:
	        for i in encodes:
	            length_list.append(len(i[0]))
	        temp=max(set(length_list), key=length_list.count)
	        encodes_temp=[]
	        for i in encodes:
	            if len(i[0])==temp:
	                encodes_temp.append(i)
	    encodes=encodes_temp
	    # encodes is a list of tuples [(encodes,score，sentences, tem)]; now encodes within encodes should have have the same length

	    last_word_sylls = len(self.dict_meters[rhyme_word][0])
	    # if the raw possible passed into it 
	    if len(templates[0])==2:
	        templates=[t[0] for t in templates]
	    else:
	        templates=templates
	    temp=[]
	    # this for loop is set so that valid_permutation_sylls does not return None
	    for t in templates:
	        if self.valid_permutation_sylls(num_sylls, t, last_word_sylls):
	            temp.append(t)
	    templates=temp

	    # sylls_list associated each template with its valid permutation, which can a lot of permutations
	    sylls_list = {' '.join(template): self.valid_permutation_sylls(num_sylls, template, last_word_sylls) for template in templates}
	    sentences={}
	    templates_length=[]

	    #record ending templates and sentences
	    record_old=[]


	    # create combined2 that associated with long templates (key) with short templates (values)
	    if mydict:
	        combined2={}
	        for k in encodes.keys():
	            combined2[k]=k.split("\n")[-1].strip()

	    valid=0
	    if mydict:
	        for template in templates:
	            for k in encodes.keys():
	                if template in mydict[combined2[k]]:
	                    valid+=1
	    for template in templates:
	        if mydict:
	            for k in encodes.keys():
	                if template in mydict[combined2[k]]:
	                    for enco in random.choices(encodes[k],k=min(len(encodes[k]),math.ceil(200/valid))):
	                        try:
	                            sentences[' '.join(enco[3]+["\n"]+template)].append((enco[2]+["\n"], enco[0], enco[1],enco[3]+["\n"]+template,template))
	                        except:
	                            sentences[' '.join(enco[3]+["\n"]+template)]=[(enco[2]+["\n"], enco[0], enco[1],enco[3]+["\n"]+template,template)]


	        else:
	            encodes_temp=random.choices(encodes, k=min(len(encodes),math.ceil(200/len(templates))))
	            for enco in encodes_temp:
	                try:
	                    sentences[' '.join(enco[3]+["\n"]+template)].append((enco[2]+["\n"], enco[0], enco[1],enco[3]+["\n"]+template,template))
	                except:
	                    sentences[' '.join(enco[3]+["\n"]+template)]=[(enco[2]+["\n"], enco[0], enco[1],enco[3]+["\n"]+template,template)]
	        templates_length.append(len(template))
	    if comment: print("how many templates before gpt sentence generation: ",len(templates_length))
	    assert len(templates_length)!=0 ,"No more templates here"

	    # now sentences are dictionary that looks like [(words, encodes, scores, templates, template)], keys are combined_templates
	    #create combined as a dict such that the combined templates are keys, and last templates are values.
	    combined={}
	    for t in templates:
	        for k in sentences.keys():
	            if sentences[k][0][4]==t:
	                try:
	                    combined[" ".join(t)].append(k)
	                except:
	                    combined[" ".join(t)]=[k]


	    for i in range(max(templates_length)):
	        sent_temp=[]
	        templates_string=[" ".join(t) for t in templates]
	        for k in sentences.keys():
	            for p in sentences[k]:
	                sent_temp.append(" ".join(p[4]))
	        sent_temp=set(sent_temp)
	        templates=set(templates_string)
	        templates=templates.intersection(sent_temp)
	        templates=[t.split() for t in list(templates)]
	        templates_length_curr=[len(t) for t in templates]
	        if max(templates_length_curr)<=i:
	            if comment: print("------------------no more sentences to parse-------------------")
	            if comment: print(i)
	            if comment: print("------------------no more sentences to parse-------------------")
	            break
	        if comment: print("---------------------templates----------------------------------------------")
	        if comment: print(templates)
	        if comment: print("----------------------sentences---------------------------------------------")
	        for keys in sentences.keys():
	            for sen in sentences[keys]:
	                if verbose: print(sen)
	        if comment: print("--------------------------------------------------------------------")
	        #pdb.set_trace()
	        context_token=[]
	        #template_to_interval={}
	        interval_to_template={}
	        interval_list=[]
	        # record the templates that are ending in this round
	        end_templates=[]
	        for t in templates:
	            if len(t)>i:
	                for k in combined[" ".join(t)]:
	                    temp=len(context_token)
	                    try:
	                        context_token+=[s[1] for s in sentences[k]]
	                        if comment: print("----------------success-----------------------------")
	                        if comment: print(t)
	                        if comment: print("----------------------success-------------------------")
	                    except:
	                        if comment: print("---------------------failure------------------------")
	                        if comment: print(t)
	                        if comment: print("--------------------failure---------------------------")



	                    #template_to_interval[t]=(temp,len(context_token))
	                    interval_to_template['-'.join((str(temp),str(len(context_token))))]=(k,t)
	                    interval_list.append((temp,len(context_token)))
	            if len(t)==i+1:
	                end_templates.append(t)
	                if comment: print("----------------------ending this round---------------------------")
	                if comment: print(t)
	                if comment: print("----------------------ending this round---------------------------")



	        m=len(context_token)
	        context_token=np.array(context_token).reshape(m,-1)
	        print("========================================== iteration =====================================================================")
	        print(len(context_token))
	        print("========================================== iteration =====================================================================")
	        new_sentences=[]
	        print("gpt before")
	        logits = score_model(model_name=self.model_name, context_token = context_token)
	        print("gpt after")
	        for interval in interval_list:
	            sp , template=interval_to_template['-'.join([str(interval[0]),str(interval[1])])]
	            POS=template[i]
	            if POS=='.' or POS==',':
	                for k,j in enumerate(np.arange(interval[0],interval[1],1)):
	                    new_sentences.append((sentences[sp][k][0]+[POS],sentences[sp][k][1]+[self.enc.encode(POS)[0]],sentences[sp][k][2],sentences[sp][k][3],template))
	            elif template in end_templates:
	                for k,j in enumerate(np.arange(interval[0],interval[1],1)):
	                    index=self.enc.encode(" "+rhyme_word)[0]
	                    record_old.append((sentences[sp][k][0] + [rhyme_word],sentences[sp][k][1] + [index],
	                        sentences[sp][k][2] + np.log(logits[j][index]),sentences[sp][k][3],template))

	            else:
	                for k,j in enumerate(np.arange(interval[0],interval[1],1)):
	                # this is processing the logits outputs for each sentences within each interval
	                # len(logits) should be len(context_token)
	                    sorted_index=np.argsort(-1*logits[j])
	                    break_point=0
	                    for index in sorted_index:
	                    	word = self.enc.decode([index]).lower().strip()
	                    	if len(word.lower().strip()) == 0:
	                    		fit_pos = False
	                    	else:
	                    		fit_pos = POS in self.words_to_pos[word]
	                    	if fit_pos:
	                    		if i==len(template)-1 and word!=rhyme_word:
	                    			continue
	                    		if word not in self.dict_meters or len(self.dict_meters[word][0]) != sylls_list[' '.join(template)][i]:
	                    			continue
	                    		new_sentences.append((sentences[sp][k][0] + [word],sentences[sp][k][1] + [index],sentences[sp][k][2] + np.log(logits[j][index]),sentences[sp][k][3],template))
	                    		break_point+=1
	                    		if break_point>=thresh_hold:
	                    			break
	        if len(new_sentences)==0 and len(record_old)==0:
	            raise ValueError("No Sentences for template:",templates," rhyme_word:",rhyme_word," num_sylls:",num_sylls)
	        print("new_sentences generated")
	        # this function records the sentences apart from the ones that are ending, which we exclude from discussion. 
	        new_sentences_temp=[]
	        sentences_temp=defaultdict(list)
	        sentences={}
	        repeat=[]
	        for item in new_sentences:
	            if item[4] not in end_templates:
	                new_sentences_temp.append(item)
	        if len(new_sentences_temp)>0:
	        	print(len(new_sentences_temp))
	        	for i,sent in enumerate(new_sentences_temp):
	        		if i%10000==0: print(i)
	        		if sent[2] not in repeat:
	        			repeat.append(sent[2])
	        			sentences_temp[' '.join(sent[3])].append((sent[0],sent[1],sent[2],sent[3],sent[4]))
	        	for key in sentences_temp.keys(): 
	        		temp=heapq.nlargest(min(len(sentences_temp[key]), math.ceil(search_space/len(sentences_temp.keys()))), sentences_temp[key], key=lambda x: x[2]/len(x[4]))
	        		sentences[key]=temp 
	        else:
	            record_old_return={}
	            if comment: print("-------------------------------------------------")
	            if comment: print(i)
	            if comment: print("Everything is recorded in record_old")
	            if comment: print("--------------------------------------------------")
	            for sent in record_old:
	                if sent[2] not in repeat:
	                    repeat.append(sent[2])
	                    try:
	                        sentences_temp[' '.join(sent[3])].append((sent[0],sent[1],sent[2],sent[4]))
	                    except:
	                        sentences_temp[' '.join(sent[3])]=[(sent[0],sent[1],sent[2],sent[4])]
	            for key in sentences_temp.keys():
	                temp=heapq.nlargest(min(len(sentences_temp[key]),search_space), sentences_temp[key], key=lambda x: x[2]/len(x[3]))
	                record_old_return[key]=temp
	            break

	    return record_old_return
def gen_line_gpt_andre_multi_template(self,encodes, num_sylls,templates=None, rhyme_word=None, search_space=100,comment=True,mydict=False,verbose=False, thresh_hold=10):
	    """
	    Uses GPT to generate multiple lines using all feasible templates conditioned upon the number of syllabus constraint, the last word cosntraint, and the also conditioned
	    on the templates passed into it, the encodes passed into it. 

	    Parameters
	    ----------
	    encodes : dict or list of tuples
	        encodes is a list of tuples [(encodes,score，sentences, tem)], where encodes is gpt2-byte encodes numbers; score is log likelihood of the prompt (encodes) passed
	        into it; sentences are a list of words directly associtated with gpt2-byte encodes numbers. tem is the last template, for example, if the previous sentences is line4, then tem is the line4 template
	        encodes can also be a dict, whose values are the same list of tuples, but whose keys are " ".join(whole templates), for example, the templates of first line+"\n" +second line +"\n"+third line
	    
	    num_sylls : num
	        num_sylls is the number of sullabus permissible for this line, usually, line1,line2,line5 has syllabus count 9, line3, lin4 has syllabus count 6

	    templates : list
	        list of templates that satisfy the syllabus constraint and last word POS constraint, given usually by self.there_is_template\

	    rhyme_word : str
	        the rhyme_word is the last word of the sentences, all sentences produced by this function will have the rhyme_word as the last word, it's usually from the story_line
	    
	    search_space: num
	        how wide the beam of the beam search is, this is directly related to how much memory the GPU has, usually search_space is set to 100-200

	    comment: Bool
	        turning this to False would suppress all comments, default True
	    mydict: Bool or dict
	        unless the sentece is last line of the limericks, whhose templates we suspect to be closely related to the templated of the fourth line, mydict is False,
	        otherwise, mydict is a dictionary that maps each fourth line template, with all its possible 5line templates.
	    debug: Bool
	        for dev, you can use this if statement to do a pdb.set_trace(), to debug

	    Returns
	    -------
	    record_old_return: dict
	        a dict whose keys are combined_long templates, and values (sentences, encodes, scores,tem)
	    """

	    # for now the rhyme_word is a word, but soon, it can be a list of words. 
	   



	    
	    # check if the encodes have the same legnth, if not, first now we go with the legnth with the most elements
	    length_list=[]
        for i in encodes:
            length_list.append(len(i[0]))
        temp=max(set(length_list), key=length_list.count)
        encodes_temp=[]
        for i in encodes:
            if len(i[0])==temp:
                encodes_temp.append(i)
	    encodes=encodes_temp
	    # encodes is a list of tuples [(encodes,score，sentences, tem)]; now encodes within encodes should have have the same length

	    last_word_sylls = len(self.dict_meters[rhyme_word][0])
	    # if the raw possible passed into it 
	    if len(templates[0])==2:
	        templates=[t[0] for t in templates]
	    else:
	        templates=templates
	    temp=[]
	    # this for loop is set so that valid_permutation_sylls does not return None
	    for t in templates:
	        if self.valid_permutation_sylls(num_sylls, t, last_word_sylls):
	            temp.append(t)
	    templates=temp

	    # sylls_list associated each template with its valid permutation, which can a lot of permutations
	    sylls_list = {' '.join(template): self.valid_permutation_sylls(num_sylls, template, last_word_sylls) for template in templates}
	    sentences={}
	    templates_length=[]

	    #record ending templates and sentences
	    record_old=[]


	    # create combined2 that associated with long templates (key) with short templates (values)
	    if mydict:
	        combined2={}
	        for k in encodes.keys():
	            combined2[k]=k.split("\n")[-1].strip()

	    valid=0
	    if mydict:
	        for template in templates:
	            for k in encodes.keys():
	                if template in mydict[combined2[k]]:
	                    valid+=1
	    for template in templates:
	        if mydict:
	            for k in encodes.keys():
	                if template in mydict[combined2[k]]:
	                    for enco in random.choices(encodes[k],k=min(len(encodes[k]),math.ceil(200/valid))):
	                        try:
	                            sentences[' '.join(enco[3]+["\n"]+template)].append((enco[2]+["\n"], enco[0], enco[1],enco[3]+["\n"]+template,template))
	                        except:
	                            sentences[' '.join(enco[3]+["\n"]+template)]=[(enco[2]+["\n"], enco[0], enco[1],enco[3]+["\n"]+template,template)]


	        else:
	            encodes_temp=random.choices(encodes, k=min(len(encodes),math.ceil(200/len(templates))))
	            for enco in encodes_temp:
	                try:
	                    sentences[' '.join(enco[3]+["\n"]+template)].append((enco[2]+["\n"], enco[0], enco[1],enco[3]+["\n"]+template,template))
	                except:
	                    sentences[' '.join(enco[3]+["\n"]+template)]=[(enco[2]+["\n"], enco[0], enco[1],enco[3]+["\n"]+template,template)]
	        templates_length.append(len(template))
	    if comment: print("how many templates before gpt sentence generation: ",len(templates_length))
	    assert len(templates_length)!=0 ,"No more templates here"

	    # now sentences are dictionary that looks like [(words, encodes, scores, templates, template)], keys are combined_templates
	    #create combined as a dict such that the combined templates are keys, and last templates are values.
	    combined={}
	    for t in templates:
	        for k in sentences.keys():
	            if sentences[k][0][4]==t:
	                try:
	                    combined[" ".join(t)].append(k)
	                except:
	                    combined[" ".join(t)]=[k]


	    for i in range(max(templates_length)):
	        sent_temp=[]
	        templates_string=[" ".join(t) for t in templates]
	        for k in sentences.keys():
	            for p in sentences[k]:
	                sent_temp.append(" ".join(p[4]))
	        sent_temp=set(sent_temp)
	        templates=set(templates_string)
	        templates=templates.intersection(sent_temp)
	        templates=[t.split() for t in list(templates)]
	        templates_length_curr=[len(t) for t in templates]
	        if max(templates_length_curr)<=i:
	            if comment: print("------------------no more sentences to parse-------------------")
	            if comment: print(i)
	            if comment: print("------------------no more sentences to parse-------------------")
	            break
	        if comment: print("---------------------templates----------------------------------------------")
	        if comment: print(templates)
	        if comment: print("----------------------sentences---------------------------------------------")
	        for keys in sentences.keys():
	            for sen in sentences[keys]:
	                if verbose: print(sen)
	        if comment: print("--------------------------------------------------------------------")
	        #pdb.set_trace()
	        context_token=[]
	        #template_to_interval={}
	        interval_to_template={}
	        interval_list=[]
	        # record the templates that are ending in this round
	        end_templates=[]
	        for t in templates:
	            if len(t)>i:
	                for k in combined[" ".join(t)]:
	                    temp=len(context_token)
	                    try:
	                        context_token+=[s[1] for s in sentences[k]]
	                        if comment: print("----------------success-----------------------------")
	                        if comment: print(t)
	                        if comment: print("----------------------success-------------------------")
	                    except:
	                        if comment: print("---------------------failure------------------------")
	                        if comment: print(t)
	                        if comment: print("--------------------failure---------------------------")



	                    #template_to_interval[t]=(temp,len(context_token))
	                    interval_to_template['-'.join((str(temp),str(len(context_token))))]=(k,t)
	                    interval_list.append((temp,len(context_token)))
	            if len(t)==i+1:
	                end_templates.append(t)
	                if comment: print("----------------------ending this round---------------------------")
	                if comment: print(t)
	                if comment: print("----------------------ending this round---------------------------")



	        m=len(context_token)
	        context_token=np.array(context_token).reshape(m,-1)
	        print("========================================== iteration =====================================================================")
	        print(len(context_token))
	        print("========================================== iteration =====================================================================")
	        new_sentences=[]
	        print("gpt before")
	        logits = score_model(model_name=self.model_name, context_token = context_token)
	        print("gpt after")
	        for interval in interval_list:
	            sp , template=interval_to_template['-'.join([str(interval[0]),str(interval[1])])]
	            POS=template[i]
	            if POS=='.' or POS==',':
	                for k,j in enumerate(np.arange(interval[0],interval[1],1)):
	                    new_sentences.append((sentences[sp][k][0]+[POS],sentences[sp][k][1]+[self.enc.encode(POS)[0]],sentences[sp][k][2],sentences[sp][k][3],template))
	            elif template in end_templates:
	                for k,j in enumerate(np.arange(interval[0],interval[1],1)):
	                    index=self.enc.encode(" "+rhyme_word)[0]
	                    record_old.append((sentences[sp][k][0] + [rhyme_word],sentences[sp][k][1] + [index],
	                        sentences[sp][k][2] + np.log(logits[j][index]),sentences[sp][k][3],template))

	            else:
	                for k,j in enumerate(np.arange(interval[0],interval[1],1)):
	                # this is processing the logits outputs for each sentences within each interval
	                # len(logits) should be len(context_token)
	                    sorted_index=np.argsort(-1*logits[j])
	                    break_point=0
	                    for index in sorted_index:
	                    	word = self.enc.decode([index]).lower().strip()
	                    	if len(word.lower().strip()) == 0:
	                    		fit_pos = False
	                    	else:
	                    		fit_pos = POS in self.words_to_pos[word]
	                    	if fit_pos:
	                    		if i==len(template)-1 and word!=rhyme_word:
	                    			continue
	                    		if word not in self.dict_meters or len(self.dict_meters[word][0]) != sylls_list[' '.join(template)][i]:
	                    			continue
	                    		new_sentences.append((sentences[sp][k][0] + [word],sentences[sp][k][1] + [index],sentences[sp][k][2] + np.log(logits[j][index]),sentences[sp][k][3],template))
	                    		break_point+=1
	                    		if break_point>=thresh_hold:
	                    			break
	        if len(new_sentences)==0 and len(record_old)==0:
	            raise ValueError("No Sentences for template:",templates," rhyme_word:",rhyme_word," num_sylls:",num_sylls)
	        print("new_sentences generated")
	        # this function records the sentences apart from the ones that are ending, which we exclude from discussion. 
	        new_sentences_temp=[]
	        sentences_temp=defaultdict(list)
	        sentences={}
	        repeat=[]
	        for item in new_sentences:
	            if item[4] not in end_templates:
	                new_sentences_temp.append(item)
	        if len(new_sentences_temp)>0:
	        	print(len(new_sentences_temp))
	        	for i,sent in enumerate(new_sentences_temp):
	        		if i%10000==0: print(i)
	        		if sent[2] not in repeat:
	        			repeat.append(sent[2])
	        			sentences_temp[' '.join(sent[3])].append((sent[0],sent[1],sent[2],sent[3],sent[4]))
	        	for key in sentences_temp.keys(): 
	        		temp=heapq.nlargest(min(len(sentences_temp[key]), math.ceil(search_space/len(sentences_temp.keys()))), sentences_temp[key], key=lambda x: x[2]/len(x[4]))
	        		sentences[key]=temp 
	        else:
	            record_old_return={}
	            if comment: print("-------------------------------------------------")
	            if comment: print(i)
	            if comment: print("Everything is recorded in record_old")
	            if comment: print("--------------------------------------------------")
	            for sent in record_old:
	                if sent[2] not in repeat:
	                    repeat.append(sent[2])
	                    try:
	                        sentences_temp[' '.join(sent[3])].append((sent[0],sent[1],sent[2],sent[4]))
	                    except:
	                        sentences_temp[' '.join(sent[3])]=[(sent[0],sent[1],sent[2],sent[4])]
	            for key in sentences_temp.keys():
	                temp=heapq.nlargest(min(len(sentences_temp[key]),search_space), sentences_temp[key], key=lambda x: x[2]/len(x[3]))
	                record_old_return[key]=temp
	            break

	    return record_old_return