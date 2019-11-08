import tensorflow as tf
import nltk
from py_files.Limericks_new2 import Limerick_Generate_new
import fire
import pdb
'''
def limericks_generation_gpt(model_name="345M",model_dir='gpt2/models/345M',search_space=50,prompt="life", retain_space=5, prob_threshold=-10):
	lg = Limerick_Generate_new()
	lg.gen_poem_andre_new(prompt=prompt,search_space=search_space, retain_space=retain_space, prob_threshold=prob_threshold)
if __name__ == '__main__':
    fire.Fire(limericks_generation_gpt)
'''
if __name__=="__main__":
	lg = Limerick_Generate_new()
	w1s_rhyme_dict, w3s_rhyme_dict= lg.get_two_sets_new_henry("queen", n_w1=2000, n_w3=2000)
	for k in w1s_rhyme_dict:
		print(k)
		print(w1s_rhyme_dict[k])
	for k in w3s_rhyme_dict:
		print(k)
		print(w3s_rhyme_dict)