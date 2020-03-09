#import tensorflow as tf
import numpy as np
import collections
from collections import defaultdict
import os
import random
import math
from math import exp
import pdb
#from gpt2.src.score import score_model
#from gpt2.src.encoder import get_encoder
import pickle
from datetime import datetime as dt
def init_parser():
    parser = argparse.ArgumentParser(description='hahahha')
    parser.add_argument('--product', '-p', default='microwave', type=str, dest='product')
    return parser.parse_args()
def softmax(x):
	if len(x)>0:
		ret=[xx/sum(x) for xx in x]
	else:
		ret=[]
	return ret

def run(args):
	with open("py_files/{}.pickle".format(args.product),"rb") as f:
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
	with open("s2f_{}.pickle".format(args.product),"wb") as f:
		pickle.dump(s2f, f)
	positive_feature_score=[]
	negative_feature_score=[]
	unique_features=[]
	for item in s2f:
		if item[2]==1:
			feature_list=[feature[0] for feature in item[1].items()]
			feature_prob=[feature[1] for feature in item[1].items()]
			feature_score=softmax(feature_prob)
			if len(feature_score)==0: continue
			for i,feature in enumerate(feature_list):
				unique_features.append(feature)
				positive_feature_score.append([feature, feature_score[i],item[3],item[4],item[5]])
		if item[2]==-1:
			feature_list=[feature[0] for feature in item[1].items()]
			feature_prob=[feature[1] for feature in item[1].items()]
			feature_score=softmax(feature_prob)
			if len(feature_score)==0: continue
			for i,feature in enumerate(feature_list):
				unique_features.append(feature)
				negative_feature_score.append([feature, -1*feature_score[i],item[3],item[4],item[5]])
	with open("positive_feature_score_{}.pickle".format(args.product),"wb") as f:
		pickle.dump(positive_feature_score, f)
	with open("negative_feature_score_{}.pickle".format(args.product),"wb") as f:
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
				text="/".join([str(item[3]),str(item[4]),str(item[2])])
				text=dt.strptime(text, '%m/%d/%Y')
				text=dt.strftime(text,'%m/%d/%Y')
				mydict_negative[feature][text].append(item[1])
		for date in mydict_negative[feature].keys():
			mydict_negative[feature][date]=[np.mean(mydict_negative[feature][date]),len(mydict_negative[feature][date])]
	with open("mydict_negative_{}.pickle".format(args.product),"wb") as f:
		pickle.dump(mydict_negative, f)
	with open("mydict_positive_{}.pickle".format(args.product),"wb") as f:
		pickle.dump(mydict_positive, f)
	bob_delta(mydict_negative,"-",args.product)
	bob_delta(mydict_positive,"+",args.product)

def bob_delta(data,sign,product):
	bobdict=defaultdict(list)
	with open("py_files/unique_dates_{}.pickle".format(product),"rb") as f:
		unique_dates=pickle.load(f)
		unique_dates=[dt.strptime(d,'%m/%d/%Y') for d in unique_dates]
		unique_dates.sort()
	for feature in data.keys():
		time=[dt.strptime(d,'%m/%d/%Y') for d in data[feature].keys()]
		time.sort()
		delta=[]
		value=[]
		count=[]
		for i in range(len(unique_dates)):
			if unique_dates[i] not in time:
				delta.append(float('NaN'))
			elif unique_dates[i] in time and time.index(unique_dates[i])==0:
				delta.append(0)
			else:
				delta.append((time[time.index(unique_dates[i])]-time[time.index(unique_dates[i])-1]).days)
		for i in range(len(unique_dates)):
			if unique_dates[i] not in time:
				value.append(float('NaN'))
				count.append(float('NaN'))
			else:
				index=time.index(unique_dates[i])
				curr=dt.strftime(time[index],'%m/%d/%Y')
				pdb.set_trace()
				value.append(data[feature][curr][0])
				count.append(data[feature][curr][1])
		bobdict[feature]=[unique_dates,delta,value,count]
	
	if sign=="+":
		with open("bobdict_positive_{}.pickle".format(product),"wb") as f:
			pickle.dump(bobdict,bobdict_positive)
	if sign=="-":
		with open("bobdict_negative_{}.pickle".format(product),"wb") as f:
			pickle.dump(bobdict,bobdict_negative)









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
	return s2f
if __name__ == '__main__':
	args=init_parser()
	run(args)
	'''
	with open("mydict_negative.pickle","rb") as f:
		mydict_negative=pickle.load(f)
	with open("mydict_positive.pickle","rb") as f:
		mydict_positive=pickle.load(f)

	bob_delta(mydict_negative,"-")
	bob_delta(mydict_positive,"+")
	'''






