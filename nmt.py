# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""

import math
import model
import numpy as np
import os
import pickle
import sys
import time
import torch

from collections import namedtuple
from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from torch.autograd import Variable
from typing import Any, Dict, List, Set, Tuple, Union
from tqdm import tqdm

from utils import batch_iter, read_corpus
from vocab import Vocab, VocabEntry


Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(object):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        super(NMT, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        src_vocab_size = len(self.vocab.src.word2id)
        tgt_vocab_size = len(self.vocab.tgt.word2id)

        self.encoder = model.EncoderRNN(vocab_size=src_vocab_size,
                                        embed_size=self.embed_size,
                                        hidden_size=self.hidden_size)
        self.decoder = model.DecoderRNN(embed_size=self.embed_size,
                                        hidden_size=self.hidden_size,
                                        output_size=tgt_vocab_size)
        self.encoder.cuda()
        self.decoder.cuda() 

        self.criterion = torch.nn.CrossEntropyLoss()

    def __call__(self, src_sents: List[List[str]], tgt_sents: List[List[str]]) -> torch.Tensor:
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of 
        target sentences.

        Args:
            src_sents: list of source sentence tokens
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`

        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the 
                log-likelihood of generating the gold-standard target sentence for 
                each example in the input batch
        """
        src_encodings, decoder_init_state = self.encode(src_sents)
        scores = self.decode(src_encodings, decoder_init_state, tgt_sents)

        return scores

    def encode(self, src_sents: List[List[str]]) -> Tuple[torch.Tensor, Any]:
        """
        Use a GRU/LSTM to encode source sentences into hidden states

        Args:
            src_sents: list of source sentence tokens

        Returns:
            src_encodings: hidden states of tokens in source sentences, this could be a variable 
                with shape (batch_size, source_sentence_length, encoding_dim), or in orther formats
            decoder_init_state: decoder GRU/LSTM's initial state, computed from source encodings
        """
        # Numberize the source sentences
        numb_src_sents = self.vocab.src.words2indices(src_sents)

        # Pad each sentence to the maximum length
        max_len = len(numb_src_sents[0])
        padded_src_sent = [sent + [0]*(max_len - len(sent)) for sent in numb_src_sents]

        # Get the original sentence lengths
        input_lengths = [len(sent) for sent in numb_src_sents]

        # Construct a long tensor (seq_len * batch_size)
        input_tensor = Variable(torch.LongTensor(padded_src_sent).t()).cuda()

        # Call encoder
        src_encodings, decoder_init_state = self.encoder(input_tensor, input_lengths)

        return src_encodings, decoder_init_state

    def decode(self, src_encodings: torch.Tensor, decoder_init_state: Any, tgt_sents: List[List[str]]) -> torch.Tensor:
        """
        Given source encodings, compute the log-likelihood of predicting the gold-standard target
        sentence tokens

        Args:
            src_encodings: hidden states of tokens in source sentences
            decoder_init_state: decoder GRU/LSTM's initial state
            tgt_sents: list of gold-standard target sentences, wrapped by `<s>` and `</s>`

        Returns:
            scores: could be a variable of shape (batch_size, ) representing the 
                log-likelihood of generating the gold-standard target sentence for 
                each example in the input batch
        """
        # TODO: for now ignoring source encodings, must use for attention

        # Numberize the target sentences
        numb_tgt_sents = self.vocab.tgt.words2indices(tgt_sents)

        # Pad each sentence to the maximum length
        max_len = max([len(sent) for sent in numb_tgt_sents])
        padded_tgt_sent = [sent + [0]*(max_len - len(sent)) for sent in numb_tgt_sents]

        # Get the original sentence lengths
        input_lengths = [len(sent) for sent in numb_tgt_sents]

        # Construct a long tensor (seq_len * batch_size)
        input_tensor = Variable(torch.cuda.LongTensor(padded_tgt_sent).t())
        
        scores = torch.zeros(input_tensor[0].size())
        last_hidden = decoder_init_state
        for t in range(1,max_len):
          # Get output from the decoder
          output, last_hidden = self.decoder(last_hidden, input_tensor[t-1].unsqueeze(0))
          # output = output.cuda()

          # Compute scores and add them
          new_scores = [self.criterion(output[:,i].float(), input_tensor[t,i].unsqueeze(0)) 
                     * (0 if t >= input_lengths[i] else 1)
                     for i in range(len(input_tensor[t]))]
          scores += torch.stack(new_scores)

        # Normalize each score by the length of the sentence, add up, normalize by batch size
        # normalizers = torch.FloatTensor(input_lengths)
        # normalizers = normalizers.cuda()
        return (scores / torch.Tensor(input_lengths).mean()) # / normalizers.mean())

    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """
        Given a single source sentence, perform beam search

        Args:
            src_sent: a single tokenized source sentence
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """

    # Greedy Decoding for testing

    #     src, dec_init_state = self.encode([src_sent])
    #     previous_word = '<sos>'

    #     greedy_ouput = []
        
    #     for _ in range(max_decoding_time_step):

    #         word_indices = self.vocab.tgt.word2indices([previous_word])            
    #         scores, dec_init_state = self.decoder(dec_init_state, word_indices)

    #         # greedy decoding
    #         max_score_word = self.vocab.tgt.word2id[scores.index(max(scores))]
    #         beam_list.append(max_score_word)

    #         # update previous word
    #         previous_word = max_score_word 

    #     # return beam_list

    #     # Beam search decoding
    #     hypotheses = {'sos': 0}  # string vs the log likelihood

    #     for t in range(max_decoding_time_step):

    #         for x in hypotheses:
    #             src, dec_init_state = self.encode([x])
	# 	          word_indices = self.vocab.tgt.word2indices(x)
    #         	  scores, dec_init_state = self.decoder(dec_init_state, word_indices)

    #             top_scores = sorted(scores, reverse=True)[:beam_size]
                
    #             for i in top_scores:
    #                 word = self.vocab.tgt.id2word[scores.index[i]]
    #                 hypotheses[x + word] = hypotheses[x] + i  
        	
    #    	  # Prune the hypotheses for the next step
    #         hypotheses = sorted(hypotheses.items(), key=lambda x: -x[1])[:beam_size] 

	# return namedtuple('Hypothesis', hypotheses.keys())(**hypotheses)
    pass 
        

    def evaluate_ppl(self, dev_data: List[Any], batch_size: int=32):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of dev sentences
            batch_size: batch size
        
        Returns:
            ppl: the perplexity on dev sentences
        """

        cum_loss = 0.
        cum_tgt_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`

        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -self.model(src_sents, tgt_sents).sum()

            cum_loss += loss
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

        return ppl

    @staticmethod
    def load(model_path: str):
        """
        Load a pre-trained model

        Returns:
            model: the loaded model
        """
        model = torch.load(model_path)
        return model

    def save(self, model_path: str):
        """
        Save current model to file
        """
        torch.save(self.model, model_path)


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score


def train(args: Dict[str, str]):
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    lr = float(args['--lr'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    vocab = pickle.load(open(args['--vocab'], 'rb'))

    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab)
    # model.cuda() or model = model.cuda() or model = NMT().cuda() # error: model has no attribute cuda

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    # Define an Adam optimizer
    optim = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=lr)

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            # Zero out the gradients
            optim.zero_grad()

            train_iter += 1

            batch_size = len(src_sents)

            # (batch_size)
            loss = model(src_sents, tgt_sents)
            print(loss.size())
            report_loss += loss.item()
            cum_loss += loss.item()

            # TODO: ensure that this can actually be called
            loss.backward()

            # Clip gradient norms
            torch.nn.utils.clip_grad_norm(list(model.encoder.parameters()) + list(model.decoder.parameters()), clip_grad)

            # Do a step of the optimizer
            optim.step()

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cumulative_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss,
                                                                                         math.exp(report_loss*batch_size*log_every / report_tgt_words),
                                                                                         cumulative_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cumulative_examples,
                                                                                         np.exp(cum_loss / cumulative_tgt_words),
                                                                                         cumulative_examples), file=sys.stderr)

                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = model.evaluate_ppl(dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # You may also save the optimizer's state
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay learning rate, and restore from previously best checkpoint
                        lr = lr * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        model_save_path

                        print('restore parameters of the optimizers', file=sys.stderr)
                        # You may also need to load the state of the optimizer saved before

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    #was_training = model.training

    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

        hypotheses.append(example_hyps)

    return hypotheses


def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results. 
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)
    if os.path.exists(args['MODEL_PATH']):
        model = NMT.load(args['MODEL_PATH'])
    else:
        model = NMT(256, 256, pickle.load(open('data/vocab.bin', 'rb')))

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def main():
    args = docopt(__doc__)

    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(args['--seed'])
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()
