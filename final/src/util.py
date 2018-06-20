import numpy as np

def padLines(lines, value, maxlen):
    maxlinelen = 0
    for i, s in enumerate(lines):
        maxlinelen = max(len(s), maxlinelen)
    for i, s in enumerate(lines):
        lines[i] = (['_'] * max(0, maxlinelen - len(s)) + s)[-maxlen:]
    return lines

def transformByWord2Vec(lines, w2v, size):
    for i, s in enumerate(lines):
        for j, w in enumerate(s):
            if w in w2v.wv:
                lines[i][j] = w2v.wv[w]
            else:
                lines[i][j] = np.zeros((1,size))
