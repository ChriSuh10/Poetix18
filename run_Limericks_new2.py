import tensorflow as tf
import nltk
from py_files.Limericks_new2 import Limerick_Generate_new
import fire
import pdb
def limericks_generation_gpt(model_name="345M",model_dir='gpt2/models/345M',search_space=100,prompt="life", thresh_hold=10):
	lg = Limerick_Generate_new()
	lg.gen_poem_andre_new(prompt=prompt,search_space=search_space,thresh_hold=thresh_hold)
if __name__ == '__main__':
    fire.Fire(limericks_generation_gpt)