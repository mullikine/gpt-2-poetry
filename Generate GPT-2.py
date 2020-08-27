#!/usr/bin/env python
# coding: utf-8

import json
import os
import numpy as np
import tensorflow as tf
import model, sample, encoder

# !ln -s ../models models # hack to make models "appear" in two places

model_name = '117M'
seed = None
nsamples = 10
batch_size = 10
length = 40
temperature = 0.8 # 0 is deterministic
top_k = 40 # 0 means no restrictions

assert nsamples % batch_size == 0

enc = encoder.get_encoder(model_name)
hparams = model.default_hparams()
with open(os.path.join('models', model_name, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))

if length is None:
    length = hparams.n_ctx // 2
elif length > hparams.n_ctx:
    raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

sess = tf.InteractiveSession()

# replace with this in script:
# with tf.Session(graph=tf.Graph()) as sess:

context = tf.placeholder(tf.int32, [batch_size, None])
np.random.seed(seed)
tf.set_random_seed(seed)
output = sample.sample_sequence(
    hparams=hparams, length=length,
    context=context,
    batch_size=batch_size,
    temperature=temperature, top_k=top_k
)

saver = tf.train.Saver()
ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
saver.restore(sess, ckpt)

from utils.list_all_files import *
import unicodedata
import os, re, random

mapping = {
 '\xa0': ' ',
 'Æ': 'AE',
 'æ': 'ae',
 'è': 'e',
 'é': 'e',
 'ë': 'e',
 'ö': 'o',
 '–': '-',
 '—': '-',
 '‘': "'",
 '’': "'",
 '“': '"',
 '”': '"'
}

def remove_special(text):
    return ''.join([mapping[e] if e in mapping else e for e in text])

def strip_word(word):
    word = re.sub('^\W*|\W*$', '', word).lower()
    return word

basenames = []
all_poems = {}
total_lines = 0
words = set()
for fn in list_all_files('../../scraping/poetry/output'):
    with open(fn) as f:
        original = open(fn).read()
        text = remove_special(original).split('\n')
        poem = text[3:]
        basename = os.path.basename(fn)
        basename = os.path.splitext(basename)[0]
        basenames.append(basename)
        all_poems[basename] = {
            'url': text[0],
            'title': text[1],
            'author': text[2],
            'poem': poem
        }
        total_lines += len(poem)
        poem = '\n'.join(poem)
        words.update([strip_word(e) for e in poem.split()])
words.remove('')
words = list(words)

print(total_lines)

def titlecase_word(word):
    return word[0].upper() + word[1:]

titlecase_word("carpenter's"), "carpenter's".title()

def random_chunk(array, length):
    start = random.randint(0, max(0, len(array) - length - 1))
    return array[start:start+length]

def random_item(array):
    return array[random.randint(0, len(array) - 1)]

random_chunk(all_poems[basenames[0]]['poem'], 2), titlecase_word(random_item(words))

seeds = '''
blue
epoch
ethereal
ineffable
iridescent
nefarious
oblivion
quiver
solitude
sonorous
'''.split()
len(seeds)

from utils.progress import progress

def clean(text):
    return text.split('<|endoftext|>')[0]

def generate(inspiration, seed):
    inspiration = remove_special(inspiration).strip()
    seed = titlecase_word(seed).strip()

    raw_text = inspiration + '\n' + seed
    context_tokens = enc.encode(raw_text)
    n_context = len(context_tokens)

    results = []
    for _ in range(nsamples // batch_size):
        out = sess.run(output, feed_dict={
            context: [context_tokens for _ in range(batch_size)]
        })
        for sample in out:
            text = enc.decode(sample[n_context:])
            result = seed + text
            results.append(result)

    return results

inspiration_lines = 16

all_results = {}
for seed in seeds:
    print(seed)
    cur = {}
    for basename in basenames:
        inspiration = random_chunk(all_poems[basename]['poem'], inspiration_lines)
        inspiration = '\n'.join(inspiration)
        results = generate(inspiration, seed)
        cur[basename] = results
    all_results[seed] = cur

import json
with open('poems.json', 'w') as f:
    json.dump(all_poems, f, separators=(',', ':'))

with open('generated.json', 'w') as f:
    json.dump(all_results, f, separators=(',', ':'))

