#!/usr/bin/env python
# coding: utf-8

# This parses poetry from www.poetryfoundation.org into files that have a title on the first line, author on the second, and the rest is the poem.

from bs4 import BeautifulSoup as bs
from utils.list_all_files import *
import hashlib

get_ipython().system('mkdir -p output')

def get_hash(text):
    return hashlib.md5(text.encode('utf8')).hexdigest()

conditions = ['.o-article .c-feature-hd', '.c-txt_attribution a', '.o-poem']
for fn in list_all_files('www.poetryfoundation.org/'):
    with open(fn) as f:
        html = f.read()
        soup = bs(html, 'html.parser')
        results = [soup.select(e) for e in conditions]
        if all(results):
            title = results[0][0].text.strip().split('\n')[0]
            author = results[1][0].text.strip().split('\n')[0]
            poem = results[2][0].get_text('\n').strip().split('\n')
            poem = [e.strip() for e in poem if len(e.strip())]
            poem = '\n'.join(poem)
            output_fn = 'output/' + get_hash(title + author) + '.txt'

            if len(poem) < 100:
                print(f'Parsing error: {fn}')
                continue

            with open(output_fn, 'w') as ff:
                ff.write('\n'.join([fn, title, author, poem]))


