"""
Part 1: Simple baseline that only uses word statistics to predict tags
python3 mp4.py --train data/brown-training.txt --test data/brown-dev.txt --algorithm baseline
"""


def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
	    test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
	    E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    wordmap = {}
    tagmap = {}
    # training
    for sentence in train:
        print(sentence)
        for pair in sentence:
            word = pair[0]
            tag = pair[1]
            if word not in wordmap:
                wordmap[word] = {}
            if tag not in wordmap[word]:
                wordmap[word][tag] = 1
            else:
                wordmap[word][tag] += 1
            if tag not in tagmap:
                tagmap[tag] = 1
            else:
                tagmap[tag] += 1

    most_freq_tag = max(tagmap, key=tagmap.get)
    result = []
    # testing
    for sentence in test:
        test_sentence = []
        for word in sentence:
            if word not in wordmap:
                test_sentence.append((word, most_freq_tag))
            else:
                tag = max(wordmap[word], key = wordmap[word].get)
                test_sentence.append((word, tag))
        result.append(test_sentence)
    return result
