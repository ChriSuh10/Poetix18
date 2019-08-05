from py_files.Limericks import Limerick_Generate
import argparse
parser = argparse.ArgumentParser(description='This file is used to test limerick production.')
parser.add_argument("seed", help="seed word used to start poem generation")
parser.add_argument("search_space", help="search space for the GPT2 model method",
                    type=int)

args = parser.parse_args()
lg = Limerick_Generate(model_name='345M',load_poetic_vectors=False)
poem = lg.gen_poem_gpt(args.seed, args.seed,
       prompt_length=100, search_space=args.search_space, story_line=True,
       enforce_syllables = True, enforce_stress = True,
       search_space_coef=[1,1,0.9,0.9])
for line in poem:
    print(' '.join(line))
