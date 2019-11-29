#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

from .model import *
from .encoder import *
from .sample import sample_sequence
import nltk.data

def create_syll_dict():
    """
    Using the cmudict file, returns a dictionary mapping words to their
    intonations (represented by 1's and 0's). Assumed to be larger than the
    corpus of words used by the model.

    Parameters
    ----------
    fname : str
        The name of the file containing the mapping of words to their
        intonations.
    """
    with open('py_files/saved_objects/cmudict-0.7b.txt', encoding='UTF-8') as f:
        lines = [line.rstrip("\n").split() for line in f if (";;;" not in line)]
        dict_meters = {}
        for i in range(len(lines)):
            line = lines[i]
            newLine = [line[0].lower()]
            if("(" in newLine[0] and ")" in newLine[0]):
                newLine[0] = newLine[0][:-3]
            chars = ""
            for word in line[1:]:
                for ch in word:
                    if(ch in "012"):
                        if(ch == "2"):
                            chars += "1"
                        else:
                            chars += ch
            newLine += [chars]
            lines[i] = newLine
            if(newLine[0] not in dict_meters):  # THIS IF STATEMENT ALLOWS FOR MULTIPLE PRONUNCIATIONS OF A WORD
                dict_meters[newLine[0]] = [chars]
            else:
                if(chars not in dict_meters[newLine[0]]):
                    dict_meters[newLine[0]] += [chars]
        dict_meters[','] = ['']
        dict_meters['.'] = ['']
        return dict_meters


def poem_is_valid(text, tokenizer, syll_dict):
    text = tokenizer.tokenize(text.replace(",",".").replace(";","."))
    if len(text) < 4:
        return False
    for i in range(4):
        curr_sylls = [4, 7] if (i == 1 or i == 2) else [7, 10]
        syll_count = 0
        for word in text[i].split():
            stripped_word = re.sub("[^a-zA-Z]+", "", word.lower().strip())
            if stripped_word not in syll_dict:
                return False
            possible_syllables = syll_dict[stripped_word]
            word_length = min(len(s) for s in possible_syllables)
            syll_count += word_length
        if syll_count < curr_sylls[0] or syll_count > curr_sylls[1]:
            return False
    for i in range(4):
        print(text[i])
    return True

def generate_poem_naive(
    model_name='117M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    raw_text=None
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    syll_dict = create_syll_dict()
    enc = get_encoder(model_name)
    hparams = default_hparams()
    try:
        with open(os.path.join('gpt2/models', model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))
    except:
        with open(os.path.join('models', model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('gpt2/models', model_name))
        saver.restore(sess, ckpt)

        context_tokens = enc.encode(raw_text)
        generated = 0
        valid = 0
        res = []
        while valid < nsamples:
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i])
                print("=" * 40 + " SAMPLE " + str(generated) + " VALID " + str(valid) + " " + "=" * 40)
                if poem_is_valid(text, tokenizer, syll_dict):
                    text = tokenizer.tokenize(text.replace(",",".").replace(";","."))[:4]
                    res.append(text)
                    valid += 1
        print("=" * 80)
        return res
