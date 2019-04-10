import json
import os
import numpy as np
import tensorflow as tf
import fire

from .model import *
from .encoder import *

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )

def score_model(
    model_name='117M',
    seed=None,
    nsamples=1,
    length=None,
    temperature=1,
    top_k=0,
    context_token=[]
):
    enc = get_encoder(model_name)
    hparams = default_hparams()
    try:
        with open(os.path.join('gpt2/models', model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))
    except:
        with open(os.path.join('models', model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

    with tf.Session(graph=tf.Graph()) as sess:

        context = tf.placeholder(tf.int32, [len(context_token), None],name="context")
        lm_output = model(hparams=hparams, X=context, past=None, reuse=tf.AUTO_REUSE)
        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        logits = logits[:, -1, :]  / tf.to_float(temperature)
        logits = top_k_logits(logits, k=top_k)
        logits = tf.nn.softmax(logits, axis=1)

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('gpt2/models', model_name))
        saver.restore(sess, ckpt)

        out = sess.run(logits, feed_dict={
            context: context_token
        })
    return out

def score_line(
    line,
    model_name='117M',
    seed=None,
    nsamples=1,
    length=None,
    temperature=1,
    top_k=0
):
    enc = get_encoder(model_name)
    hparams = default_hparams()
    with open(os.path.join('gpt2/models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    encoded_context = [enc.encode(line)]
    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [1, None],name="context")
        lm_output = model(hparams=hparams, X=context, past=None, reuse=tf.AUTO_REUSE)
        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        logits = tf.nn.log_softmax(logits, axis=2)
    #         logits = logits[:, -1, :]  / tf.to_float(temperature)
    #         logits = top_k_logits(logits, k=top_k)
    #         logits = tf.nn.softmax(logits, axis=1)

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('gpt2/models', model_name))
        saver.restore(sess, ckpt)

        out = sess.run(logits, feed_dict={
            context: encoded_context
        })
    log_prob = 0
    for i, e in enumerate(encoded_context[0]):
        log_prob += out[0, i, e]
    return log_prob / len(encoded_context[0])

if __name__ == '__main__':
    #fire.Fire(score_model)
    fire.Fire(score_model2)
