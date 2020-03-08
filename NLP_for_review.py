import tensorflow as tf
import numpy as np
from gensim.parsing.preprocessing import remove_stopwords
import collections
from collections import defaultdict
import tqdm
import os
import re
import random
import pickle
import math
from math import exp
import pdb
from gpt2.src.score import score_model
from gpt2.src.encoder import get_encoder
import pickle

def softmax(x):
	ret=[xx for xx in x]
	ret=[r/sum(ret) for r in ret]
	assert sum(ret)==1, "softmax wrong"
	return ret

def run():
	with open("py_files/hair_dryer.pickle","rb") as f:
		data=pickle.load(f)
	data_grouped=[]
	for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]:
		temp=[]
		if i<16:
			for d in data:
				if len(d[0])==i:
					temp.append(d)
		if i==16:
			for d in data:
				if len(d[0])>=i:
					temp.append(d)
		if len(temp)==0:
			continue
		data_grouped.append(temp)
	s2f=[]
	for ii,datadata in enumerate(data_grouped):
		print("******************************** Starts Batch {} **********************************".format(ii))
		s2f=run_batch(datadata,s2f)
		print("******************************** Finishes Batch {} **********************************".format(ii))
	with open("s2f.pickle","wb") as f:
		pickle.dump(s2f, f)
	positive_feature_score=[]
	negative_feature_score=[]
	unique_features=[]
	for item in s2f:
		if item[2]==1:
			feature_list=[feature[0] for feature in item[1].items()]
			feature_prob=[feature[1] for feature in item[1].items()]
			feature_score=softmax(feature_prob)
			for i,feature in enumerate(feature_list):
				unique_features.append(feature)
				positive_feature_score.append([feature, feature_score[i],item[3],item[4],item[5]])
		if item[2]==-1:
			feature_list=[feature[0] for feature in item[1].items()]
			feature_prob=[feature[1] for feature in item[1].items()]
			feature_score=softmax(feature_prob)
			for i,feature in enumerate(feature_list):
				unique_features.append(feature)
				negative_feature_score.append([feature, -1*feature_score[i],item[3],item[4],item[5]])
	with open("positive_feature_score.pickle","wb") as f:
		pickle.dump(positive_feature_score, f)
	with open("negative_feature_score.pickle","wb") as f:
		pickle.dump(negative_feature_score, f)
	unique_features=set(unique_features)
	mydict_positive=defaultdict(list)
	for feature in unique_features:
		mydict_positive[feature]=defaultdict(list)
		for item in positive_feature_score:
			if item[0]==feature:
				mydict_positive[feature]["/".join([str(item[3]),str(item[4]),str(item[2])])].append(item[1])
		for date in mydict_positive[feature].keys():
			mydict_positive[feature][date]=[np.mean(mydict_positive[feature][date]),len(mydict_positive[feature][date])]
	mydict_negative=defaultdict(list)
	for feature in unique_features:
		mydict_negative[feature]=defaultdict(list)
		for item in negative_feature_score:
			if item[0]==feature:
				mydict_negative[feature]["/".join([str(item[3]),str(item[4]),str(item[2])])].append(item[1])
		for date in mydict_negative[feature].keys():
			mydict_negative[feature][date]=[np.mean(mydict_negative[feature][date]),len(mydict_negative[feature][date])]
	with open("mydict_negative.pickle","wb") as f:
		pickle.dump(mydict_negative, f)
	with open("mydict_positive.pickle","wb") as f:
		pickle.dump(mydict_positive, f)






def encodes_align(previous_data):
	encodes_length=[len(i) for i in previous_data]
	encodes=[i[-min(encodes_length):] for i in previous_data]
	return encodes


def split_chunks(data):
	data_list=[]
	chuck_len = 1000
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

def run_batch(data,s2f):
	enc = get_encoder("345M")
	context_token=[enc.encode(d[0]) for d in data]
	features=[d[2] for d in data]
	context_token_list=split_chunks(encodes_align(context_token))
	feature_list=split_chunks(features)
	assert len(feature_list)==len(context_token_list), "feature token context token size mismatch"
	for i, context_token in enumerate(context_token_list):
		m=len(context_token)
		context_token=np.array(context_token).reshape(m,-1)
		print("******************************** gpt2 Starts Iteration {} **********************************".format(i))
		logits = score_model(model_name="345M", context_token = context_token)
		print("******************************** gpt2 Finished Iteration {} **********************************".format(i))
		for j, logit in enumerate(logits):
			temp_dict=defaultdict(list)
			for feature in feature_list[i][j]:
				temp=[]
				feature_token=enc.encode(feature)
				for token in feature_token:
					temp.append(logit[token])
				temp_dict[feature]=max(temp)
			d=data[i*1000+j]
			s2f.append([d[0],temp_dict,d[3],d[4],d[5],d[6]])
	print(s2f[0])
	return s2f
if __name__ == '__main__':
	run()






