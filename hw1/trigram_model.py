import sys
from collections import defaultdict
import math
import random
import os
import os.path

"""
COMS W4705 - Natural Language Processing
Homework 1 - Programming Component: Trigram Language Models
Yassine Benajiba
"""


def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile, 'r') as corpus:
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence


def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)


def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    ngrams = []

    if n > 1:
        sequence = (n - 1) * ['START'] + sequence + ['STOP']
    else:
        sequence = ['START'] + sequence + ['STOP']

    for i in range(0, len(sequence) + 1 - n):
        temp = []
        for j in range(n):
            temp.append(sequence[i + j].split(',')[0])
        ngrams.append(tuple(temp))
    return ngrams


class TrigramModel(object):

    def __init__(self, corpusfile):
        # Iterate through the corpus once to build a lexicon
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

        self.total_grams = sum(self.unigramcounts.values())  # included START and END
        self.total_sentence = self.unigramcounts[('START',)]

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """

        self.unigramcounts = defaultdict(int)  # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        ## Your code here

        for sentence in corpus:
            line_tri = get_ngrams(sentence, 3)
            line_bi = get_ngrams(sentence, 2)
            line_uni = get_ngrams(sentence, 1)
            # print(line_tri[0])

            for gram in line_tri:
                self.trigramcounts[gram] += 1

            for gram in line_bi:
                self.bigramcounts[gram] += 1

            for gram in line_uni:
                self.unigramcounts[gram] += 1

        return

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if trigram not in self.trigramcounts.keys():
            return 0.0

        if trigram[:2] == ('START', 'START'):
            p_tri = self.trigramcounts[trigram] / self.total_sentence
        else:
            p_tri = self.trigramcounts[trigram] / self.bigramcounts[trigram[:2]]
        return p_tri

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if bigram not in self.bigramcounts.keys():
            return 0.0  # in case division by 0

        return self.bigramcounts[bigram] / self.unigramcounts[bigram[:1]]

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        # hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.
        return self.unigramcounts[unigram] / self.total_grams

    def generate_sentence(self, t=20):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        # first_word = ["START","START"]
        return
        # return result

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation).
        """
        lambda1 = 1 / 3.0
        lambda2 = 1 / 3.0
        lambda3 = 1 / 3.0

        bigram = (trigram[1], trigram[2])
        unigram = (trigram[2],)

        smooth_p = lambda1 * self.raw_trigram_probability(trigram) + \
                   lambda2 * self.raw_bigram_probability(bigram) + \
                   lambda3 * self.raw_unigram_probability(unigram)

        return smooth_p

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigram = get_ngrams(sentence, 3)
        sum_prob = 0
        for tri in trigram:
            prob = self.smoothed_trigram_probability(tri)
            if prob == 0:
                pass
            else:
                sum_prob += math.log2(prob)

        return sum_prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6)
        Returns the log probability of an entire sequence.
        """
        sum_p, M = 0, 0
        for sentence in corpus:
            M += len(sentence)
            sum_p += self.sentence_logprob(sentence)
        l = sum_p / M
        pp = 2 ** (-l)

        return pp


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
    model1 = TrigramModel(training_file1)  # high
    model2 = TrigramModel(training_file2)  # low

    total = 0
    correct = 0

    for f in os.listdir(testdir1):
        total += 1
        pp_h1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
        pp_l1 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
        if pp_h1 < pp_l1:
            correct += 1

    for f in os.listdir(testdir2):
        total += 1
        pp_h2 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
        pp_l2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
        if pp_l2 < pp_h2:
            correct += 1

    return correct / total


if __name__ == "__main__":
    print("Argument List:", str(sys.argv))
    # model = TrigramModel(sys.argv[1])

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    # Testing perplexity:
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)

    # Essay scoring experiment:
    acc = essay_scoring_experiment('hw1_data/ets_toefl_data/train_high.txt',
                                   'hw1_data/ets_toefl_data/train_low.txt',
                                   "hw1_data/ets_toefl_data/test_high",
                                   "hw1_data/ets_toefl_data/test_low")
    print(acc)  # 85%
