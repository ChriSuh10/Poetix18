from py_files.Limericks import Limerick_Generate
lg = Limerick_Generate(model_name='345M',load_poetic_vectors=False)
poem = lg.gen_poem_gpt("mary", "mary",
       prompt_length=100, search_space=2000, story_line=True,
       enforce_syllables = True, enforce_stress = True,
       search_space_coef=[1,1,0.9,0.9])
