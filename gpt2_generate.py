from py_files.Limericks import Limerick_Generate
import argparse
parser = argparse.ArgumentParser(description='This file is used to test limerick production.')
parser.add_argument("seed", help="seed word used to start poem generation")
parser.add_argument("seed2", help="second seed word")
# parser.add_argument("search_space", help="search space for the GPT2 model method",
#                     type=int)

args = parser.parse_args()
lg = Limerick_Generate(model_name='345M',load_poetic_vectors=False)
poem = lg.gen_poem_gpt(args.seed, args.seed2,
       default_templates=
       [['WP$', 'NN', 'VBD', 'JJR', 'THAN', 'NN'],
        ['PDT', 'DT', 'NNS', 'VBD', 'IN'],
        ['CC', 'DT', 'NN', 'MD', 'VB', 'VBN'],
        ['SO', 'PRP', 'VBD', 'TO', 'NN']
       ],
       prompt_length=0, search_space=50,
       enforce_syllables = True, enforce_stress = True,
       search_space_coef=[1,1,0.9,0.8])
for line in poem:
    print(' '.join(line))
