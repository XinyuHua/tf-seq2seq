from __future__ import print_function
from __future__ import division

from collections import *
import re, os

DATA_PATH = '/data/xinyu/cmv/counterarg/question_generation/tmp/'
SEQ2SEQ_PATH = '/data/xinyu/cmv/counterarg/question_generation/trainable/tf-seq2seq/'

_PAD = '_PAD'
_GO = '_GO'
_END = '_END'
_UNK = '_UNK'
PAD_ID = 0
GO_ID = 1
END_ID = 2
UNK_ID = 3
SPECIAL_TOKENS = [(_PAD, PAD_ID), (_GO, GO_ID),  (_END, END_ID), (_UNK, UNK_ID)]

_DIGIT_RE = re.compile(br"\d")

def build_vocab():
    src_counter = Counter()
    tgt_counter = Counter()
    for line in open(DATA_PATH + 'train.root.tokenized'):
        lines = line.strip().split('\t')
        src = lines[0].split()
        tgt = lines[1].split()
        for token in src:
            word = _DIGIT_RE.sub(b"0", token)
            src_counter[token] += 1
        for token in tgt:
            word = _DIGIT_RE.sub(b"0", token)
            tgt_counter[token] += 1
    ordered_src = OrderedDict(sorted(src_counter.items(), key=lambda t: t[1], reverse=True))
    fout_src = open(SEQ2SEQ_PATH + 'vocab.src', 'w')
    for src_token in ordered_src:
        fout_src.write(src_token + '\t' + str(src_counter[src_token]) + '\n')
    fout_src.close()
    ordered_tgt = OrderedDict(sorted(tgt_counter.items(), key=lambda t: t[1], reverse=True))
    fout_tgt = open(SEQ2SEQ_PATH + 'vocab.tgt', 'w')
    for tgt_token in ordered_tgt:
        fout_tgt.write(tgt_token + '\t' + str(tgt_counter[tgt_token]) + '\n')
    fout_tgt.close()
    print('raw vocab src: %d' % len(src_counter))
    print('raw vocab tgt: %d' % len(tgt_counter))


class dataPreparer(object):
    def __init__(self):
        return

    def load_vocab(self, src_path, tgt_path, src_size=50000, tgt_size=50000):
        self.src_vocab = dict()
        self.tgt_vocab = dict()
        for t in SPECIAL_TOKENS:
            self.src_vocab[t[0]] = t[1]
            self.tgt_vocab[t[0]] = t[1]
        src_idx = len(self.src_vocab)
        for line in open(src_path):
            token = line.strip().split('\t')[0]
            self.src_vocab[token] = src_idx
            src_idx += 1
            if src_idx == src_size:
                break
        tgt_idx = len(self.tgt_vocab)
        for line in open(tgt_path):
            token = line.strip().split('\t')[0]
            self.tgt_vocab[token] = tgt_idx
            tgt_idx += 1
            if tgt_idx == tgt_size:
                break
        print('vocab loaded')

    def make_data(self, in_path, out_path):
        dataset = os.path.basename(in_path)
        dataset = dataset.split('.')[0]
        print('making data %s from %s to %s' % (dataset, in_path, out_path))
        fout_src = open(out_path + 'src-' + dataset + '.ids', 'w')
        fout_tgt = open(out_path + 'tgt-' + dataset + '.ids', 'w')
        for line in open(in_path).readlines():
            src = line.strip().split('\t')[0].split()
            tgt = line.strip().split('\t')[1].split()
            for token in src:
                token = _DIGIT_RE.sub(b"0", token)
                cur_id = self.src_vocab[token] if token in self.src_vocab else UNK_ID
                fout_src.write(str(cur_id) + " ")
            fout_src.write("\n")
            fout_tgt.write(str(GO_ID) + " ")
            for token in tgt:
                token = _DIGIT_RE.sub(b"0", token)
                cur_id = self.tgt_vocab[token] if token in self.tgt_vocab else UNK_ID
                fout_tgt.write(str(cur_id) + " ")
            fout_tgt.write(str(END_ID) + "\n")
        fout_src.close()
        fout_tgt.close()


def main():
    dp = dataPreparer()
    dp.load_vocab(src_path = SEQ2SEQ_PATH + "vocab.src", tgt_path = SEQ2SEQ_PATH + "vocab.tgt")
    dp.make_data(in_path = DATA_PATH + "train.root.tokenized", out_path = SEQ2SEQ_PATH)
    dp.make_data(in_path = DATA_PATH + "valid.root.tokenized", out_path = SEQ2SEQ_PATH)

if __name__=='__main__':
    main()