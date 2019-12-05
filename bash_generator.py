def bash_generator():
	type="no_story"
	saved_directory="final_testing_DTS_no_story_no_diversity"
	search_space=20
	retain_space=30
	word_embedding_coefficient=0.1
	mode="multi"
	diversity=False
	cuda=1
	prompt_list="love, cunning, dog, blood, death, war, disease, world, planet, fire, water, sports, love, car, animal, violent, opera, monster, library, market, noble, doctor, funeral, ball, body, smart, exercise, gun, art, music, boxing, forest, philosophy, night, scary, creativity, evil, angry, pride, law, school, light, rich, color, leader, park, airplane, loss, weight, useful, applaud, home, union, child, working, cheat, fall, time, hope, flower, random, impressive"
	prompt_list=set(prompt_list.split(", "))
	with open("run_Limericks_"+saved_directory+"_"+str(cuda)+".sh","w") as f:
		f.write("#!/bin/bash \n")
		for prompt in prompt_list:
			sentence="CUDA_VISIBLE_DEVICES="+str(cuda)+"   python3 run_Limericks.py"+" --type "+str(type)+" --saved_directory "+str(saved_directory)+" --mode "+str(mode)+" --diversity "+ str(diversity)+ " --cuda "+str(cuda)+" --word_embedding_coefficient "+str(word_embedding_coefficient)+" --search_space "+str(search_space)+" --retain_space "+str(retain_space)+" --prompt "+ str(prompt)+"\n"
			f.write(sentence)
if __name__ == '__main__':
    bash_generator()