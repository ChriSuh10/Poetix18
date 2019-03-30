import json
import os
import numpy as np
import tensorflow as tf
import fire

import model, sample, encoder

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
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
):

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    raw_text = input("Model prompt >>> ")
    while not raw_text:
        print('Prompt should not be empty!')
        raw_text = input("Model prompt >>> ")
    context_tokens = enc.encode(raw_text)
    with tf.Session(graph=tf.Graph()) as sess:

        context = tf.placeholder(tf.int32, [batch_size, None])
        lm_output = model.model(hparams=hparams, X=context, past=None, reuse=tf.AUTO_REUSE)
        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        logits = logits[:, -1, :]  / tf.to_float(temperature)
        logits = top_k_logits(logits, k=top_k)
        samples = tf.multinomial(logits, num_samples=5, output_dtype=tf.int32, seed=100)

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        out = sess.run(samples, feed_dict={
            context: [context_tokens for _ in range(batch_size)]
        })
        # print(out)
        print(enc.decode(out[0]))

if __name__ == '__main__':
    fire.Fire(score_model)
