def bash_generator():
	type="no_story"
	saved_directory="final_testing_no_story"
	search_space=100
	retain_space=3
	word_embedding_coefficient=0.1
	mode="multi"
	diversity=True
	cuda=2
	prompt_list="blood, death, war, queen, happy, world, planet, fire, water, game, love, vegetable, fish, theater, tiger, library, fairy, duke, print, click"
	prompt_list=prompt_list.split(", ")
	with open("run_Limericks"+"_"+str(cuda)+".sh","w") as f:
		f.write("#!/bin/bash \n")
		for prompt in prompt_list:
			sentence="CUDA_VISIBLE_DEVICES="+str(cuda)+"   python3 run_Limericks.py"+" --type "+str(type)+" --saved_directory "+str(saved_directory)+" --mode "+str(mode)+" --diversity "+ str(diversity)+ " --cuda "+str(cuda)+" --word_embedding_coefficient "+str(word_embedding_coefficient)+" --search_space "+str(search_space)+" --retain_space "+str(retain_space)+" --prompt "+ str(prompt)+"\n"
			f.write(sentence)
if __name__ == '__main__':
    bash_generator()