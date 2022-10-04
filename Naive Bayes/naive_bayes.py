# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# python3 mp1.py --lowercase LOWERCASE --stemming STEMMING --bigram BIGRAM --laplace 0.005 --bigram_laplace 0.005 --bigram_lambda 0.5 --pos_prior 0.5
from curses import curs_set
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


"""
load_data calls the provided utility to load in the dataset.
You can modify the default values for stemming and lowercase, to improve performance when
    we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def print_paramter_vals(laplace, pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

# count all the positive and negative unigram(words) in the training set
def count_unigrams(train_set, train_labels):
    pos_unigram_count = {}
    neg_unigram_count = {}
    for i in range(len(train_labels)):
        if train_labels[i] == 0:    # neg
            for word in train_set[i]:
                if word not in neg_unigram_count:
                    neg_unigram_count[word] = 1
                else:
                    neg_unigram_count[word] += 1
        else:   # pos
            for word in train_set[i]:
                if word not in pos_unigram_count:
                    pos_unigram_count[word] = 1
                else:
                    pos_unigram_count[word] += 1
    return pos_unigram_count, neg_unigram_count


def naiveBayes(train_set, train_labels, dev_set, laplace=0.0055, pos_prior=0.5, silently=False):
    print_paramter_vals(laplace,pos_prior)

    # count positive and negative words
    positive_word_count, negative_word_count = count_unigrams(train_set, train_labels)

    positive_total = sum(positive_word_count.values())
    negative_total = sum(negative_word_count.values())
    v_pos = len(positive_word_count)
    v_neg = len(negative_word_count)

    # calculate positive and negative posterior probabilities for each review
    review_labels = []
    for dev in dev_set:
        post_pos_total = 0
        post_neg_total = 0
        for word in dev:
            count_word_pos = positive_word_count[word] if word in positive_word_count else 0
            count_word_neg = negative_word_count[word] if word in negative_word_count else 0
            # log transform
            post_pos = math.log((count_word_pos + laplace) / (positive_total + laplace * (v_pos + 1)))
            post_neg = math.log((count_word_neg + laplace) / (negative_total + laplace * (v_neg + 1)))
            post_pos_total += post_pos
            post_neg_total += post_neg
        
        # add prior probabilities
        post_pos_total += math.log(pos_prior)
        post_neg_total += math.log(1 - pos_prior)

        # compare positive and negative posterior probabilities
        if post_pos_total < post_neg_total:
            review_labels.append(0)
        else:
            review_labels.append(1)
    return review_labels



def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def count_bigrams(train_set, train_labels):
    pos_bigram_count = {}
    neg_bigram_count = {}
    for i in range(len(train_labels)):
        cur_set = train_set[i]
        if train_labels[i] == 0:
            for j in range(1, len(cur_set)):
                bigram = (cur_set[j-1], cur_set[j])
                if bigram not in neg_bigram_count:
                    neg_bigram_count[bigram] = 1
                else:
                    neg_bigram_count[bigram] += 1
        else:
            for k in range(1, len(cur_set)):
                bigram = (cur_set[k-1], cur_set[k])
                if bigram not in pos_bigram_count:
                    pos_bigram_count[bigram] = 1
                else:
                    pos_bigram_count[bigram] += 1
    return pos_bigram_count, neg_bigram_count


# main function for the bigrammixture model
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.005, bigram_laplace=0.005, bigram_lambda=0.5, pos_prior=0.8, silently=False):
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    pos_unigram_count, neg_unigram_count = count_unigrams(train_set, train_labels)
    pos_unigram_total = sum(pos_unigram_count.values())
    neg_unigram_total = sum(neg_unigram_count.values())
    v_pos_unigram = len(pos_unigram_count)
    v_neg_unigram = len(neg_unigram_count)

    pos_bigram_count, neg_bigram_count = count_bigrams(train_set, train_labels)
    print(pos_bigram_count)
    pos_bigram_total = sum(pos_bigram_count.values())
    neg_bigram_total = sum(neg_bigram_count.values())
    v_pos_bigram = len(pos_bigram_count)
    v_neg_bigram = len(neg_bigram_count)

    review_labels = []
    for dev in dev_set:
        pos_unigram_sum = 0
        neg_unigram_sum = 0
        pos_bigram_sum = 0
        neg_bigram_sum = 0
        # unigram part
        for word in dev:
            count_pos_unigram = pos_unigram_count[word] if word in pos_unigram_count else 0
            count_neg_unigram = neg_unigram_count[word] if word in neg_unigram_count else 0
            pos_unigram_sum += math.log((count_pos_unigram + unigram_laplace) / (pos_unigram_total + unigram_laplace*(v_pos_unigram + 1)))
            neg_unigram_sum += math.log((count_neg_unigram + unigram_laplace) / (neg_unigram_total + unigram_laplace*(v_neg_unigram + 1)))
        pos_unigram_sum += math.log(pos_prior)
        neg_unigram_sum += math.log(1 - pos_prior)

        # bigram part
        for i in range(1, len(dev)):
            bigram = (dev[i-1], dev[i])
            count_pos_bigram = pos_bigram_count[bigram] if bigram in pos_bigram_count else 0
            count_neg_bigram = neg_bigram_count[bigram] if bigram in neg_bigram_count else 0
            pos_bigram_sum += math.log((count_pos_bigram + bigram_laplace) / (pos_bigram_total + bigram_laplace*(v_pos_bigram + 1)))
            neg_bigram_sum += math.log((count_neg_bigram + bigram_laplace) / (neg_bigram_total + bigram_laplace*(v_neg_bigram + 1)))
        pos_bigram_sum += math.log(pos_prior)
        neg_bigram_sum += math.log(1 - pos_prior)
        
        pos_posterior = (1 - bigram_lambda) * pos_unigram_sum + bigram_lambda * pos_bigram_sum
        neg_posterior = (1 - bigram_lambda) * neg_unigram_sum + bigram_lambda * neg_bigram_sum
        if pos_posterior > neg_posterior:
            review_labels.append(1)
        else:
            review_labels.append(0)
    return review_labels