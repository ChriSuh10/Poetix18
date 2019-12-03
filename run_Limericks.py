import tensorflow as tf
import nltk
import fire
import pdb
def limericks_generation_gpt(model_name="345M",model_dir='gpt2/models/345M',search_space=50,prompt="life", retain_space=5, prob_threshold=None, embedding=0.0, type="original"):
	if type=="original":
		from py_files.Limericks_original import Limerick_Generate_new
	if type=="34linked":
		from py_files.Limericks_34_linked_temp import Limerick_Generate_new
	lg = Limerick_Generate_new()
	lg.gen_poem_andre_new(prompt=prompt,search_space=search_space, retain_space=retain_space, prob_threshold=prob_threshold, word_embedding_coefficient=embedding)
if __name__ == '__main__':
    fire.Fire(limericks_generation_gpt)
