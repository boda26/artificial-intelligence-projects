"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
python3 mp4.py --train data/brown-training.txt --test data/brown-dev.txt --algorithm viterbi_1
"""
import math

def count_tag_word(train):
    tag_map = {}
    tag_word_map = {}
    for sentence in train:
        for word, tag in sentence:
            tag_map[tag] = tag_map[tag] + 1 if tag in tag_map else 1
    for t in tag_map:
        tag_word_map[t] = {}
        for sentence in train:
            for word, tag in sentence:
                if tag == t:
                    tag_word_map[t][word] = tag_word_map[t][word] + 1 if word in tag_word_map[t] else 1
    return tag_map, tag_word_map

def count_tag_pair(train, tag_map):
    tag_pair_map = {}
    for tag in tag_map:
        tag_pair_map[tag] = {}
        for sentence in train:
            for i in range(1, len(sentence)):
                if sentence[i-1][1] == tag:
                    next_tag = sentence[i][1]
                    tag_pair_map[tag][next_tag] = tag_pair_map[tag][next_tag] + 1 if next_tag in tag_pair_map[tag] else 1
    return tag_pair_map

def get_em_probs(tag_map, tag_word_map, laplace):
    em_probs = {}
    for tag in tag_map:
        em_probs[tag] = {}
        words = tag_word_map[tag]
        v = len(words)
        n = 0
        for key, value in words.items():
            n += value
        for word, count_words in words.items():
            em_probs[tag][word] = math.log(laplace + count_words) - math.log(n + laplace * (v + 1))
        em_probs[tag]['UNKNOWN'] = math.log(laplace) - math.log(n + laplace * (1 + v))
    return em_probs

def get_trans_probs(tag_map, tag_pair_map, laplace):
    trans_probs = {}
    for tag in tag_map:
        trans_probs[tag] = {}
        next_tags = tag_pair_map[tag]
        v = len(next_tags)
        n = 0
        for key, value in next_tags.items():
            n += value
        for next_tag, count_next_tag in next_tags.items():
            trans_probs[tag][next_tag] = math.log(laplace + count_next_tag) - math.log(n + laplace * (v + 1))
    return trans_probs

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tag_map, tag_word_map = count_tag_word(train)
    tag_pair_map = count_tag_pair(train, tag_map)
    laplace = 0.00001
    em_probs = get_em_probs(tag_map, tag_word_map, laplace)
    trans_probs = get_trans_probs(tag_map, tag_pair_map, laplace)
    
    for sentence in test:
        trellis = []
        for i in range(len(sentence)):
            tag_set = {}
            for tag in tag_map:
                tag_set[tag] = 0
            trellis.append(tag_set)
        prev_map = {}
        
    return []