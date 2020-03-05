import tensorflow as tf
import nltk
import pdb
import os
import pickle
import numpy as np
from collections import defaultdict
import heapq
import random
import argparse

def init_parser():
    parser = argparse.ArgumentParser(description='Evaluate which epoch')
    parser.add_argument("--saved_directory",'-dir', default='testing',type=str,dest='saved_directory')
    parser.add_argument("--search_space",'-ser', default=100,type=int,dest='search_space')
    parser.add_argument("--retain_space",'-re', default=3,type=int,dest='retain_space')
    parser.add_argument("--word_embedding_coefficient",'-w', default=0.1,type=float,dest='word_embedding_coefficient')
    parser.add_argument("--gender",'-g',default="female",type=str,dest='gender')
    return parser.parse_args()

def printing(data, f, f_final, f_final_best,word_embedding_coefficient, words_to_names_rhyme_dict,f_all,prompt):
	try:
		with open(f_final+".pickle","rb") as pickle_in:
			data_old=pickle.load(pickle_in)
	except:
		with open(f_final+".pickle","wb") as pickle_in:
			data_old={"score":[],"adjusted_score":[]}
			pickle.dump(data_old,pickle_in)
	try:
		with open(f_final_best+".pickle","rb") as pickle_in:
			data_old_best=pickle.load(pickle_in)
	except:
		with open(f_final_best+".pickle","wb") as pickle_in:
			data_old_best={"score":[],"adjusted_score":[]}
			pickle.dump(data_old_best,pickle_in)
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

		f.write("======================= template: {} ============================  \n".format(t+1))
		f.write(k)
		f.write("----------------------- original sentences ------------------------------------ \n")
		for jj,j in enumerate(temp_data[k]):
			adjusted_score=np.mean(j[1])+word_embedding_coefficient*np.mean(j[5])
			score=np.mean(j[1])
			data_curr_score.append(score)
			data_curr_adjusted_score.append(adjusted_score)
			f.write("-------------------------score:  {};  adjusted_score: {}----------------------- \n".format(score, adjusted_score))
			limerick=list(j[2])
			limerick[limerick.index("\n")-1]=random.choice(words_to_names_rhyme_dict[j[4][0]])
			if jj<3:
				f_all.write("{}:{}".format(prompt,score)+"\n")
				f_all.write(" ".join(limerick)+"\n")
			f.write(" ".join(limerick))
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
	data_old_best_score=data_old_best["score"]
	data_old_best_adjusted_score=data_old_best["adjusted_score"]
	data_curr_best_score=heapq.nlargest(min(len(data_curr_score),5), data_curr_score, key=lambda x: x)
	data_curr_best_adjusted_score=heapq.nlargest(min(len(data_curr_adjusted_score),5), data_curr_score, key=lambda x: x)
	data_curr_best_score+=data_old_best_score
	data_curr_best_adjusted_score+=data_old_best_adjusted_score
	data_curr_best={"score":data_curr_best_score,"adjusted_score":data_curr_best_adjusted_score}
	data_old_score=data_old["score"]
	data_old_adjusted_score=data_old["adjusted_score"]
	data_curr_score+=data_old_score
	data_curr_adjusted_score+=data_old_adjusted_score
	data_curr={}
	data_curr["score"]=data_curr_score
	data_curr["adjusted_score"]=data_curr_adjusted_score
	with open(f_final+".pickle","wb") as pickle_in:
		pickle.dump(data_curr,pickle_in)
	with open(f_final_best+".pickle","wb") as pickle_in:
		pickle.dump(data_curr_best,pickle_in)
def limericks_generation_gpt(model_name="345M",model_dir='gpt2/models/345M',prompt="blood",args=None):
	gender=args.gender
	saved_directory=args.saved_directory
	search_space=args.search_space
	retain_space=args.retain_space
	word_embedding_coefficient=args.word_embedding_coefficient

	from py_files.Limericks_naive import Limerick_Generate_new
	lg = Limerick_Generate_new()
	saved_directory=saved_directory
	f_final=saved_directory +"/"+"results_"+str(search_space)+"_"+str(retain_space)+"_"+str(word_embedding_coefficient)
	f_final_best=saved_directory +"/"+"best_results_"+str(search_space)+"_"+str(retain_space)+"_"+str(word_embedding_coefficient)
	f1_path=saved_directory+"/"+"success.txt"
	f2_path=saved_directory+"/"+"success.pickle"
	print("=========================================")
	print(saved_directory)
	print("=========================================")
	if saved_directory not in os.listdir(os.getcwd()):
		os.mkdir(saved_directory)
		print("==================== here ===================================")
		print(saved_directory)
	result_file_path = saved_directory +"/"+ prompt+"_" + gender + '_' +str(search_space)+"_"+str(retain_space)+"_"+str(word_embedding_coefficient)
	all_result_file_path=saved_directory +"/" + str(search_space)+"_"+str(retain_space)+"_"+str(word_embedding_coefficient)
	previous_data, words_to_names_rhyme_dict=lg.gen_poem_andre_new(gender=gender,prompt=prompt,search_space=search_space, retain_space=retain_space, word_embedding_coefficient=word_embedding_coefficient)
	print("==================== here here===================================")
	with open(result_file_path+".pickle","wb") as f3:
		pickle.dump(previous_data,f3)
	print("==================== here here here===================================")
	
	with open(result_file_path+".txt","a+") as f:
		with open(all_result_file_path+".txt","a+") as f_all:
			printing(previous_data,f, f_final,f_final_best,word_embedding_coefficient,  words_to_names_rhyme_dict,f_all,prompt)
	print("==================== here here here here===================================")
	if len(previous_data)>0:
		with open(f1_path,"a+") as f1:
			f1.write(prompt+str(search_space)+"_"+str(retain_space)+"_"+str(word_embedding_coefficient)+"\n")
		try:
			with open(f2_path,"rb") as f2:
				data=pickle.load(f2)
				data.append(prompt)
			with open(f2_path,"wb") as f2:
				pickle.dump(data,f2)
		except:
			with open(f2_path,"wb") as f2:
				pickle.dump([],f2)
		print("==================== here here here here here===================================")
	
if __name__ == '__main__':
	data1="born, shaken, restore, laugh, tears"
	#, surprise, kindness, humiliation, victory, wedding, alien, holiday, christmas, thanksgiving, birthday, injury, pillow, fiance, dawn, traffic, heartbreak, wine, beer, musuem, mountain, river, memory, mud, spider, rain, season, winter, throne, politics, promise, beach, bank, money, limerick"
	data2="love, cunning, dog, blood, death, war"
	#disease, world, planet, fire, water, sports, love, car, animal, violent, opera, monster, library, market, noble, doctor, funeral, ball, body, smart, exercise, gun, art, music, boxing, forest, philosophy, night, scary, creativity, evil, angry, pride, law, school, light, rich, color, leader, park, airplane, loss, weight, useful, applaud, home, union, child, working, cheat, fall, time, hope, flower, random, impressive"
	prompt_list=list(data1.split(", ")+data2.split(", "))
	slurm_task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
	prompt=prompt_list[slurm_task_id]
	print(prompt)
	limericks_generation_gpt(prompt=prompt,args=init_parser())
	
	
