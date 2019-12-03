import tensorflow as tf
import nltk
import fire
import pdb
def run_test(model_name="345M",model_dir='gpt2/models/345M', type="original", saved_directory="final_testing"):
	if type=="original":
		from py_files.Limericks_original import Limerick_Generate_new
	if type=="34linked":
		from py_files.Limericks_34_linked import Limerick_Generate_new
	lg = Limerick_Generate_new(model_name=model_name,model_dir=model_dir, saved_directory=saved_directory)
	prompt_list="hound, blood, death, war, queen, happy, world, planet, fire, water, game, love, vegetable, fish, theater, tiger, library, fairy, duke, print, click"
	prompt_list=prompt_list.split(", ")
	word_embedding_coefficient_list=[0.1]
	space_list=[(100,3)]
	mode_list=["multi",0,1,2,3,4,5,6,7,8,9,10,11]
	diversity_list=[True,False]
	if saved_directory not in os.listdir(os.getcwd()):
			os.mkdir(saved_directory)
	f1=open(saved_directory+"/"+"success.txt",a+)
	f2=open(saved_directory+"/"+"failure.txt",a+)
	for search_space, retain_space in space_list:
		for word_embedding_coefficient in word_embedding_coefficient_list:
			for mode in mode_list:
				if mode=="multi":
					for diversity in diversity_list:
						f_final=saved_directory +"/"+"results"+str(search_space)+"_"+str(retain_space)+"_"+str(word_embedding_coefficient)+"_"+mode+"_"+str(diversity)+"_"+"original"
						counter=0
						data_curr={"score":[],"adjusted_score":[]}
						with open(f_final+"_"+str(counter)+".pickle",wb) as pickle_in:
							pickle.dump(data_curr,pickle_in)
						for prompt in prompt_list:
							try:
								lg.gen_poem_andre_new(prompt=prompt,search_space=search_space, 
										retain_space=retain_space, word_embedding_coefficient=word_embedding_coefficient, mode=mode, diversity=diversity, f_final=f_final, counter=counter)
								counter+=1
								f1.write(prompt+str(search_space)+"_"+str(retain_space)+"_"+str(word_embedding_coefficient)+"_"+mode+"_"+str(diversity)+"_"+"original"+"\n")
							except:
								f2.write(prompt+str(search_space)+"_"+str(retain_space)+"_"+str(word_embedding_coefficient)+"_"+mode+"_"+str(diversity)+"_"+"original"+"\n")
								continue

				else:
					diversity=False
					f_final=saved_directory +"/"+"results"+str(search_space)+"_"+str(retain_space)+"_"+str(word_embedding_coefficient)+"_"+str(mode)+"_"+str(diversity)+"_"+"original"
					counter=0
					data_curr={"score":[],"adjusted_score":[]}
					with open(f_final+"_"+str(counter)+".pickle",wb) as pickle_in:
						pickle.dump(data_curr,pickle_in)
					for prompt in prompt_list:
						try:
							lg.gen_poem_andre_new(prompt=prompt,search_space=search_space, 
									retain_space=retain_space, word_embedding_coefficient=word_embedding_coefficient, mode=mode, diversity=diversity,f_final=f_final, counter=counter)
							counter+=1
							f1.write(prompt+str(search_space)+"_"+str(retain_space)+"_"+str(word_embedding_coefficient)+"_"+mode+"_"+str(diversity)+"_"+"original"+"\n")
						except:
							f2.write(prompt+str(search_space)+"_"+str(retain_space)+"_"+str(word_embedding_coefficient)+"_"+mode+"_"+str(diversity)+"_"+"original"+"\n")
							continue
if __name__ == '__main__':
    fire.Fire(limericks_generation_gpt)