import tensorflow as tf
import nltk
import fire
import pdb
import os
import pickle
import numpy as np
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
def limericks_generation_gpt(model_name="345M",model_dir='gpt2/models/345M',type="original", saved_directory="final_testing", 
	prompt="blood",search_space=100, retain_space=3, word_embedding_coefficient=0.1, 
	mode="multi", diversity=True,cuda=3):
	if type=="original":
		from py_files.Limericks_original import Limerick_Generate_new
	if type=="34linked":
		from py_files.Limericks_34_linked import Limerick_Generate_new
	if type=="no_story":
		from py_files.Limericks_no_story import Limerick_Generate_new
	lg = Limerick_Generate_new()
	saved_directory=saved_directory+str(cuda)
	f_final=saved_directory +"/"+"results"+str(search_space)+"_"+str(retain_space)+"_"+str(word_embedding_coefficient)+"_"+str(mode)+"_"+str(diversity)+"_"+str(type)
	f1_path=saved_directory+"/"+"success.txt"
	f2_path=saved_directory+"/"+"failure.txt"
	if saved_directory not in os.listdir(os.getcwd()):
			os.mkdir(saved_directory)
	result_file_path = saved_directory +"/"+ prompt+"_" + str(search_space)+"_"+str(retain_space)+"_"+str(word_embedding_coefficient)+"_"+str(mode)+"_"+str(diversity)+"_"+str(type)
	try:
		previous_data, template_to_line=lg.gen_poem_andre_new(prompt=prompt,search_space=search_space, retain_space=retain_space, 
			word_embedding_coefficient=word_embedding_coefficient, mode=mode, diversity=diversity, f_final=f_final)
		with open(result_file_path+".pickle","wb") as f3:
			pickle.dump(previous_data,f3)
		with open(result_file_path+".txt","a+") as f:
			printing(previous_data,f, f_final, word_embedding_coefficient, template_to_line)
		with open(f1_path,"a+") as f1:
			f1.write(prompt+str(search_space)+"_"+str(retain_space)+"_"+str(word_embedding_coefficient)+"_"+str(mode)+"_"+str(diversity)+"_"+str(type)+"\n")
	except:
		with open(f2_path,"a+") as f2:
			f2.write(prompt+str(search_space)+"_"+str(retain_space)+"_"+str(word_embedding_coefficient)+"_"+str(mode)+"_"+str(diversity)+"_"+str(type)+"\n")

if __name__ == '__main__':
    fire.Fire(limericks_generation_gpt)
