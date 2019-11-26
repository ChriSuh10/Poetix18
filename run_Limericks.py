import tensorflow as tf
import nltk
from py_files.Limericks_original import Limerick_Generate_new
import fire
import pdb
def limericks_generation_gpt(model_name="345M",model_dir='gpt2/models/345M',search_space=50,prompt="life", retain_space=5, prob_threshold=-10, embedding=0.0):
	lg = Limerick_Generate_new()
	lg.gen_poem_andre_new(prompt=prompt,search_space=search_space, retain_space=retain_space, prob_threshold=prob_threshold, word_embedding_coefficient=embedding)
if __name__ == '__main__':
    fire.Fire(limericks_generation_gpt)
