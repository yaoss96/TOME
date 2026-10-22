#!/usr/bin/env python3
"""Download the ten SuiteSparse graphs of the paper into data/graphs/.

    python get_graphs.py              # all ten
    python get_graphs.py email EPA    # some

Each graph comes from the SuiteSparse Matrix Collection (sparse.tamu.edu)
in Matrix Market format and is saved as data/graphs/<name>.mtx.
"""
import io
import os
import sys
import tarfile
import urllib.request

GRAPHS = {
    'email': 'Arenas', 'GD06_Java': 'Pajek', 'ODLIS': 'Pajek', 'SciMet': 'Pajek',
    'EPA': 'Pajek', 'Erdos972': 'Pajek', 'California': 'Pajek', 'FA': 'Pajek',
    'EAT_RS': 'Pajek', 'HEP-th-new': 'Pajek',
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graphs')


def fetch(name):
    dst = os.path.join(OUT, name + '.mtx')
    if os.path.exists(dst):
        print('have', dst)
        return True
    group = GRAPHS[name]
    url = f'https://suitesparse-collection-website.herokuapp.com/MM/{group}/{name}.tar.gz'
    print('fetch', url)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        data = urllib.request.urlopen(req, timeout=120).read()
    except Exception as e:  # the collection's file server is not always reachable
        print(f'  failed ({e}).  Download the Matrix Market file of {group}/{name}\n'
              f'  from https://sparse.tamu.edu/{group}/{name} and save {name}.mtx in {OUT}')
        return False
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tf:
        member = tf.getmember(f'{name}/{name}.mtx')
        with open(dst, 'wb') as f:
            f.write(tf.extractfile(member).read())
    print('wrote', dst)
    return True


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    names = sys.argv[1:] or list(GRAPHS)
    for n in names:
        if n not in GRAPHS:
            raise SystemExit(f'unknown graph {n}; known: {", ".join(GRAPHS)}')
    if not all([fetch(n) for n in names]):
        sys.exit(1)
