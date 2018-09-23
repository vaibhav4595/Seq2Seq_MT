#!/usr/bin/env python
"""
Generate the vocabulary file for neural network training
A vocabulary file is a mapping of tokens to their indices

Usage:
    vocab.py --train-src=<file> --train-tgt=<file> [options] VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --train-src=<file>         File of training source sentences
    --train-tgt=<file>         File of training target sentences
    --size=<int>               vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
    --vocab-type=<str>         type of vocab [default: 'word']
"""

from typing import List
from collections import Counter
from itertools import chain
from docopt import docopt
import pickle

from utils import read_corpus, input_transpose


class VocabEntry(object):
    def __init__(self, vocab_type='word'):
        self.word2id = dict()
        self.vocab_type = vocab_type
        self.unk_id = 3
        self.word2id['<pad>'] = 0
        self.word2id['<s>'] = 1
        self.word2id['</s>'] = 2
        self.word2id['<unk>'] = 3

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def numberize(self, sents):
        if self.vocab_type == 'word':
          return self.words2indices(sents)
        elif self.vocab_type == 'char':
          return self.char2indices(sents)
        elif self.vocab_type == 'bpe':
          return self.bpe2indices(sents)

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def char2indices(self, sents):
        def __sent2indices(sent):
          indices = []
          # Check whether the sentence starts in SOS
          if sent.startswith('<s>'):
            indices.append(self.word2id['<s>'])
            indices.append(self.word2id[' '])
            sent = ' '.join(sent.split()[1:])

          # Check whether the sentence ends in EOS
          eos_end = sent.endswith("</s>")
          if eos_end:
            sent = ' '.join(sent.split()[:-1])

          indices += [self.word2id[c] for c in sent]

          if eos_end:
            indices.append(self.word2id[' '])
            indices.append(self.word2id['</s>'])

          return indices

        if type(sents[0]) == list:
            return [__sent2indices(' '.join(s)) for s in sents]
        else:
            return __sent2indices(' '.join(sents))

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        vocab_entry = VocabEntry(vocab_type='word')

        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print(f'number of word types: {len(word_freq)}, number of word types w/ frequency >= {freq_cutoff}: {len(valid_words)}')

        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)

        return vocab_entry

    @staticmethod
    def from_corpus_char(corpus, size, freq_cutoff=2):
        vocab_entry = VocabEntry(vocab_type='char')

        char_corpus = [' '.join(sent) for sent in corpus]

        token_freq = Counter(chain(*char_corpus))
        valid_tokens = [w for w, v in token_freq.items() if v >= freq_cutoff]
        print(f'number of token types: {len(token_freq)}, number of token types w/ frequency >= {freq_cutoff}: {len(valid_tokens)}')

        top_k_tokens = sorted(valid_tokens, key=lambda w: token_freq[w], reverse=True)[:size]
        for token in top_k_tokens:
            vocab_entry.add(token)

        return vocab_entry

    @staticmethod
    def from_corpus_bpe(corpus, size, freq_cutoff=2):
        vocab_entry = VocabEntry(vocab_type='bpe')

        # Create a count of all tokens
        token_freq = Counter()
        for sent in corpus:
          for word in sent:
            # Now we have to consider all sets of letter counts from 1 to the length of the sentence
            for i in range(1, len(word)):
              # Iterate over the possible sentence starts
              for s in range(0, len(word)-i+1):
                # Update the freq of the token
                token_freq[word[s:s+i]] += 1
        
        print("Built vocab frequencies")

        # Initialize the tokens to be all unigrams
        tokens = sorted([w for w in token_freq.keys() if len(w) == 1], key=token_freq.get, reverse=True)
        token_set = set(tokens)

        # Keep token pairs
        token_pairs = set([t1+t2 for t1 in tokens for t2 in tokens])

        # Keep unselected words
        sorted_tokens = [w for w in sorted(token_freq.keys(), key=token_freq.get, reverse=True) if w not in token_set][:size]

        # Iteratively, expand by adding the combination of tokens that is the most frequent. 
        # Repeat this until we hit the desired vocab size, or below the frequency.
        while len(tokens) < size:
          if len(tokens) % 500 == 0:
            print("Vocabulary size %d reached" % len(tokens))

          # Find the token with the maximum frequency
          i = 0
          while sorted_tokens[i] not in token_pairs:
            i += 1
          next_token = sorted_tokens[i]

          # Remove new token from the list
          sorted_tokens = sorted_tokens[:i] + sorted_tokens[i+1:]

          # Break if couldn't find sufficient token
          if next_token is None or token_freq[next_token] < freq_cutoff:
            break
        
          # Otherwise add the token
          tokens.append(next_token)
          token_set.add(next_token)
        
          # Remove from pairs
          token_set.remove(next_token)

          # Add to pairs
          for t in tokens:
            token_pairs.add(t+next_token)
            token_pairs.add(next_token+t)


        print(f'number of token types: {len(tokens)}, number of token types w/ frequency >= {freq_cutoff}: {len(tokens)}')

        # Add space to vocab
        vocab_entry.add(' ')

        # Add each of the tokens to the vocab
        for token in tokens:
            vocab_entry.add(token)

        import pdb; pdb.set_trace()
        return vocab_entry


class Vocab(object):
    def __init__(self, src_sents, tgt_sents, vocab_size, freq_cutoff, vocab_type='word'):
        assert len(src_sents) == len(tgt_sents)

        print('initialize source vocabulary ..')
        if vocab_type == 'word':
          self.src = VocabEntry.from_corpus(src_sents, vocab_size, freq_cutoff)
        elif vocab_type == 'char':
          self.src = VocabEntry.from_corpus_char(src_sents, vocab_size, freq_cutoff)
        elif vocab_type == 'bpe':
          self.src = VocabEntry.from_corpus_bpe(src_sents, vocab_size, freq_cutoff)

        print('initialize target vocabulary ..')
        if vocab_type == 'word':
          self.tgt = VocabEntry.from_corpus(tgt_sents, vocab_size, freq_cutoff)
        elif vocab_type == 'char':
          self.tgt = VocabEntry.from_corpus_char(tgt_sents, vocab_size, freq_cutoff)
        elif vocab_type == 'bpe':
          self.tgt = VocabEntry.from_corpus_bpe(tgt_sents, vocab_size, freq_cutoff)

    def __repr__(self):
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))


if __name__ == '__main__':
    args = docopt(__doc__)

    print('read in source sentences: %s' % args['--train-src'])
    print('read in target sentences: %s' % args['--train-tgt'])

    src_sents = read_corpus(args['--train-src'], source='src')
    tgt_sents = read_corpus(args['--train-tgt'], source='tgt')

    vocab = Vocab(src_sents, tgt_sents, int(args['--size']), int(args['--freq-cutoff']), vocab_type=args['--vocab-type'])
    print('generated vocabulary, source %d words, target %d words' % (len(vocab.src), len(vocab.tgt)))

    pickle.dump(vocab, open(args['VOCAB_FILE'], 'wb'))
    print('vocabulary saved to %s' % args['VOCAB_FILE'])
