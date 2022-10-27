"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
python3 mp4.py --train data/brown-training.txt --test data/brown-dev.txt --algorithm viterbi_1
"""
import math

def make_map(train):
    tag_map = {}
    word_map = {}
    for sentence in train:
        for pair in sentence:
            word = pair[0]
            tag = pair[1]
            word_map[word] = word_map[word] + 1 if word in word_map else 1
            tag_map[tag] = tag_map[tag] + 1 if tag in tag_map else 1
    return tag_map, word_map

def get_em_probs(train, tag_map, word_map, laplace):
    em_probs = {}
    for tag in tag_map:
        em_probs[tag] = {}
        for word in word_map:
            em_probs[tag][word] = 0
    for sentence in train:
        for word,tag in sentence:
            em_probs[tag][word] += 1
    tag_word_map = {}
    for tag in tag_map:
        tag_word_map[tag] = 0
        for word in word_map:
            if em_probs[tag][word] != 0:
                tag_word_map[tag] += 1
    for tag in tag_map:
        for word in word_map:
            em_probs[tag][word] = math.log(laplace + em_probs[tag][word]) - math.log(tag_map[tag] + laplace * (1 + tag_word_map[tag]))
    return em_probs, tag_word_map

def get_trans_probs(train, tag_map, tag_word_map, laplace):
    trans_probs = {}
    for tag1 in tag_map:
        trans_probs[tag1] = {}
        for tag2 in tag_map:
            trans_probs[tag1][tag2] = 0
    for sentence in train:
        l1 = sentence[:-1]
        l2 = sentence[1:]
        for a, b in zip(l1, l2):
            trans_probs[a[1]][b[1]] += 1
    for tag1 in tag_map:
        for tag2 in tag_map:
            trans_probs[tag1][tag2] = math.log(laplace + trans_probs[tag1][tag2]) - math.log(tag_map[tag1] + laplace * len(tag_word_map))
    return trans_probs

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    out = []
    tag_map, word_map = make_map(train)
    ini = {}
    laplace = 0.0001
    for tag in tag_map:
        ini[tag] = 0
    for sentence in train:
        ini[sentence[1][1]] += 1
    for tag in tag_map:
        ini[tag] = math.log(ini[tag] + laplace) - math.log(len(train) + laplace * len(tag_map))
    em_probs, tag_word_map = get_em_probs(train, tag_map, word_map, laplace)
    trans_probs = get_trans_probs(train, tag_map, tag_word_map, laplace)
    # run viterbi
    for sentence in test:
        n = len(sentence)
        trellis = {}
        back = {}
        for tag in tag_map:
            back[tag] = [-1 for k in range(n)]
            trellis[tag] = [float('-inf') for k in range(n)]
        for tag in tag_map:
            if sentence[1] in em_probs[tag]:
                p1 = em_probs[tag][sentence[1]]
            else:
                p1 = math.log(laplace) - math.log(tag_map[tag] + laplace * (1 + tag_word_map[tag]))
            trellis[tag][1] = ini[tag] + p1
            back[tag][1] = 0
        for i in range(2, n):
            for tag in tag_map:
                if sentence[i] in em_probs[tag]:
                    p = em_probs[tag][sentence[i]]
                else:
                    p = math.log(laplace) - math.log(tag_map[tag] + laplace * (1 + tag_word_map[tag]))
                for tag0 in tag_map:
                    max_val = trellis[tag0][i-1] + trans_probs[tag0][tag] + p
                    if trellis[tag][i] < max_val:
                        trellis[tag][i] = max_val
                        back[tag][i] = tag0
        max_val = float('-inf')
        max_tag = 0
        for tag in tag_map:
            if trellis[tag][n-1] > max_val:
                max_val = trellis[tag][n-1]
                max_tag = tag
        sentence_out = []
        for k in range(n-1, 0, -1):
            sentence_out.append((sentence[k], max_tag))
            max_tag = back[max_tag][k]
        sentence_out.append(("START", "START"))
        sentence_out.reverse()
        out.append(sentence_out)
    return out
