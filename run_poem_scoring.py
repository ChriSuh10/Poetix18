import tensorflow as tf
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords
import collections
from collections import defaultdict
import os
import re
import pickle
from gpt2.src.score import score_model
from gpt2.src.generate_prompt import generate_prompt
from gpt2.src.encoder import get_encoder
from py_files.Limericks import Limerick_Generate
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--path")
args = parser.parse_args()

in_path = args.path
lg = Limerick_Generate(model_name='345M')
batch_size = 10
model_name = '345M'
enc = get_encoder(model_name)
in_txt = open(in_path, 'rb')
lines = [[],[],[],[],[]]

counter = 0
for line in in_txt:
    line = line.rstrip().decode('utf-8')
    if len(line) < 5:
        continue
    lines[counter].append([re.sub('[^a-zA-Z]+', '', word.lower()) for word in line.split()])
    counter += 1
    counter %= 5
poems = []
for i in range(0, len(lines[0])):
    poems.append(lines[0][i] + lines[1][i] + lines[2][i] + lines[3][i] + lines[4][i])

encodes = [[enc.encode(' '.join(poems[i])), poems[i], []] for i in range(len(poems))]
finished = []
index = 0
while True:
    new_encodes = []
    for i in range(len(encodes)):
        if index + 1 < len(encodes[i][0]):
            new_encodes.append(encodes[i])
        else:
            finished.append(encodes[i])
    encodes = new_encodes
    if len(encodes) == 0:
        break
    logits = score_model(model_name=model_name, context_token=[e[0][:index+1] for e in encodes])
    for i in range(len(encodes)):
        # Calculate logit score
            encodes[i][2].append(np.log(logits[i][encodes[i][0][index+1]]))
    index += 1

print("score generated successfully")
now = datetime.now().time() # time object
file_name = str(len(poems)) + "_poems_generated_at_" + now.strftime("%H_%M_%S") + ".p"
print("saved at " + file_name)
pickle.dump(finished, open(file_name, "wb"))
