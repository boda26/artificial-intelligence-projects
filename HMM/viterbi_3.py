"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
python3 mp4.py --train data/brown-training.txt --test data/brown-dev.txt --algorithm viterbi_3
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

def count_hapax(train, tag_map, word_map, laplace):
    hapax = {}

    X_EST = {}
    X_FUL = {}
    X_ABLE = {}
    X_EN = {}
    X_ISE = {}

    X_IVE = {}
    X_LESS = {}
    X_NESS = {}
    X_ION = {}
    X_OR = {}
    
    X_ANT = {}
    X_SHIP = {}
    X_TH = {}
    X_ITY = {}
    X_ARY = {}

    X_ING = {}
    X_LY = {}
    X_ED = {}
    X_S = {}
    X_ER = {}
    for sentence in train:
        for word, tag in sentence:
            if word_map[word] == 1:
                if word[-3:] == 'est':
                    X_EST[word] = tag
                elif word[-3:] == 'ful':
                    X_FUL[word] = tag
                elif word[-4:] == 'able':
                    X_ABLE[word] = tag
                elif word[-2:] == 'en':
                    X_EN[word] = tag
                elif word[-3:] == 'ise':
                    X_ISE[word] = tag
                elif word[-3:] == 'ive':
                    X_IVE[word] = tag
                elif word[-4:] == 'less':
                    X_LESS[word] = tag
                elif word[-4:] == 'ness':
                    X_NESS[word] = tag
                elif word[-3:] == 'ion':
                    X_ION[word] = tag
                elif word[-2:] == 'or':
                    X_OR[word] = tag
                elif word[-3:] == 'ant':
                    X_ANT[word] = tag
                elif word[-4:] == 'ship':
                    X_SHIP[word] = tag
                elif word[-2:] == 'th':
                    X_TH[word] = tag
                elif word[-3:] == 'ity':
                    X_ITY[word] = tag
                elif word[-3:] == 'ary':
                    X_ARY[word] = tag
                elif word[-3:] == 'ing':
                    X_ING[word] = tag
                elif word[-2:] == 'ly':
                    X_LY[word] = tag
                elif word[-2:] == 'ed':
                    X_ED[word] = tag
                elif word[-1:] == 's':
                    X_S[word] = tag
                elif word[-2:] == 'er':
                    X_ER[word] = tag
                else:
                    hapax[word] = tag
    est_tag_count = {}
    ful_tag_count = {}
    able_tag_count = {}
    en_tag_count = {}
    ise_tag_count = {}
    ive_tag_count = {}
    less_tag_count = {}
    ness_tag_count = {}
    ion_tag_count = {}
    or_tag_count = {}
    ant_tag_count = {}
    ship_tag_count = {}
    th_tag_count = {}
    ity_tag_count = {}
    ary_tag_count = {}
    ing_tag_count = {}
    ly_tag_count = {}
    ed_tag_count = {}
    s_tag_count = {}
    er_tag_count = {}
    hapax_tag_count = {}
    for tag in tag_map:
        est_tag_count[tag] = 0
        ful_tag_count[tag] = 0
        able_tag_count[tag] = 0
        en_tag_count[tag] = 0
        ise_tag_count[tag] = 0
        ive_tag_count[tag] = 0
        less_tag_count[tag] = 0
        ness_tag_count[tag] = 0
        ion_tag_count[tag] = 0
        or_tag_count[tag] = 0
        ant_tag_count[tag] = 0
        ship_tag_count[tag] = 0
        th_tag_count[tag] = 0
        ity_tag_count[tag] = 0
        ary_tag_count[tag] = 0
        ing_tag_count[tag] = 0
        ly_tag_count[tag] = 0
        ed_tag_count[tag] = 0
        s_tag_count[tag] = 0
        er_tag_count[tag] = 0
        hapax_tag_count[tag] = 0
    
    for word, tag in X_EST.items():
        est_tag_count[tag] += 1
    for word, tag in X_FUL.items():
        ful_tag_count[tag] += 1 
    for word, tag in X_ABLE.items():
        able_tag_count[tag] += 1 
    for word, tag in X_EN.items():
        en_tag_count[tag] += 1 
    for word, tag in X_ISE.items():
        ise_tag_count[tag] += 1
    for word, tag in X_IVE.items():
        ive_tag_count[tag] += 1
    for word, tag in X_LESS.items():
        less_tag_count[tag] += 1 
    for word, tag in X_NESS.items():
        ness_tag_count[tag] += 1 
    for word, tag in X_ION.items():
        ion_tag_count[tag] += 1
    for word, tag in X_OR.items():
        or_tag_count[tag] += 1
    for word, tag in X_ANT.items():
        ant_tag_count[tag] += 1
    for word, tag in X_SHIP.items():
        ship_tag_count[tag] += 1
    for word, tag in X_TH.items():
        th_tag_count[tag] += 1
    for word, tag in X_ITY.items():
        ity_tag_count[tag] += 1
    for word, tag in X_ARY.items():
        ary_tag_count[tag] += 1 
    for word, tag in X_ING.items():
        ing_tag_count[tag] += 1 
    for word, tag in X_LY.items():
        ly_tag_count[tag] += 1
    for word, tag in X_ED.items():
        ed_tag_count[tag] += 1
    for word, tag in X_S.items():
        s_tag_count[tag] += 1
    for word, tag in X_ER.items():
        er_tag_count[tag] += 1
    for word, tag in hapax.items():
        hapax_tag_count[tag] += 1
    
    est_probs = {}
    ful_probs = {}
    able_probs = {}
    en_probs = {}
    ise_probs = {}
    ive_probs = {}
    less_probs = {}
    ness_probs = {}
    ion_probs = {}
    or_probs = {}
    ant_probs = {}
    ship_probs = {}
    th_probs = {}
    ity_probs = {}
    ary_probs = {}
    ing_probs = {}
    ly_probs = {}
    ed_probs = {}
    s_probs = {}
    er_probs = {}
    hapax_probs = {}
    for tag in tag_map:
        est_probs[tag] = (laplace + est_tag_count[tag]) / (len(X_EST) + laplace * (1 + len(tag_map)))
        ful_probs[tag] = (laplace + ful_tag_count[tag]) / (len(X_FUL) + laplace * (1 + len(tag_map)))
        able_probs[tag] = (laplace + able_tag_count[tag]) / (len(X_ABLE) + laplace * (1 + len(tag_map)))
        en_probs[tag] = (laplace + en_tag_count[tag]) / (len(X_EN) + laplace * (1 + len(tag_map)))
        ise_probs[tag] = (laplace + ise_tag_count[tag]) / (len(X_ISE) + laplace * (1 + len(tag_map)))
        ive_probs[tag] = (laplace + ive_tag_count[tag]) / (len(X_IVE) + laplace * (1 + len(tag_map)))
        less_probs[tag] = (laplace + less_tag_count[tag]) / (len(X_LESS) + laplace * (1 + len(tag_map)))
        ness_probs[tag] = (laplace + ness_tag_count[tag]) / (len(X_NESS) + laplace * (1 + len(tag_map)))
        ion_probs[tag] = (laplace + ion_tag_count[tag]) / (len(X_ION) + laplace * (1 + len(tag_map)))
        or_probs[tag] = (laplace + or_tag_count[tag]) / (len(X_OR) + laplace * (1 + len(tag_map)))
        ant_probs[tag] = (laplace + ant_tag_count[tag]) / (len(X_ANT) + laplace * (1 + len(tag_map)))
        ship_probs[tag] = (laplace + ship_tag_count[tag]) / (len(X_SHIP) + laplace * (1 + len(tag_map)))
        th_probs[tag] = (laplace + th_tag_count[tag]) / (len(X_TH) + laplace * (1 + len(tag_map)))
        ity_probs[tag] = (laplace + ity_tag_count[tag]) / (len(X_ITY) + laplace * (1 + len(tag_map)))
        ary_probs[tag] = (laplace + ary_tag_count[tag]) / (len(X_ARY) + laplace * (1 + len(tag_map)))
        ing_probs[tag] = (laplace + ing_tag_count[tag]) / (len(X_ING) + laplace * (1 + len(tag_map)))
        ly_probs[tag] = (laplace + ly_tag_count[tag]) / (len(X_LY) + laplace * (1 + len(tag_map)))
        ed_probs[tag] = (laplace + ed_tag_count[tag]) / (len(X_ED) + laplace * (1 + len(tag_map)))
        s_probs[tag] = (laplace + s_tag_count[tag]) / (len(X_S) + laplace * (1 + len(tag_map)))
        er_probs[tag] = (laplace + er_tag_count[tag]) / (len(X_ER) + laplace * (1 + len(tag_map)))
        hapax_probs[tag] = (laplace + hapax_tag_count[tag]) / (len(hapax) + laplace * (1 + len(tag_map)))
    # print(hapax_probs)
    suffix_probs = [est_probs, ful_probs, able_probs, en_probs, ise_probs, ive_probs, less_probs, ness_probs, ion_probs, or_probs, ant_probs, ship_probs, th_probs, ity_probs, ary_probs, ing_probs, ly_probs, ed_probs, s_probs, er_probs, hapax_probs]
    return suffix_probs

def get_em_probs(train, tag_map, word_map, laplace, suffix_probs):
    est_probs = suffix_probs[0]
    ful_probs = suffix_probs[1]
    able_probs = suffix_probs[2]
    en_probs = suffix_probs[3]
    ise_probs = suffix_probs[4]
    ive_probs = suffix_probs[5]
    less_probs = suffix_probs[6] 
    ness_probs = suffix_probs[7]
    ion_probs = suffix_probs[8]
    or_probs = suffix_probs[9] 
    ant_probs = suffix_probs[10]
    ship_probs = suffix_probs[11]
    th_probs = suffix_probs[12]
    ity_probs = suffix_probs[13]
    ary_probs = suffix_probs[14]
    ing_probs = suffix_probs[15]
    ly_probs = suffix_probs[16]
    ed_probs = suffix_probs[17]
    s_probs = suffix_probs[18]
    er_probs = suffix_probs[19]
    hapax_probs = suffix_probs[20]

    hapax = {}
    X_EST = {}
    X_FUL = {}
    X_ABLE = {}
    X_EN = {}
    X_ISE = {}
    X_IVE = {}
    X_LESS = {}
    X_NESS = {}
    X_ION = {}
    X_OR = {}
    X_ANT = {}
    X_SHIP = {}
    X_TH = {}
    X_ITY = {}
    X_ARY = {}
    X_ING = {}
    X_LY = {}
    X_ED = {}
    X_S = {}
    X_ER = {}
    for sentence in train:
        for word, tag in sentence:
            if word_map[word] == 1:
                if word[-3:] == 'est':
                    X_EST[word] = tag
                elif word[-3:] == 'ful':
                    X_FUL[word] = tag
                elif word[-4:] == 'able':
                    X_ABLE[word] = tag
                elif word[-2:] == 'en':
                    X_EN[word] = tag
                elif word[-3:] == 'ise':
                    X_ISE[word] = tag
                elif word[-3:] == 'ive':
                    X_IVE[word] = tag
                elif word[-4:] == 'less':
                    X_LESS[word] = tag
                elif word[-4:] == 'ness':
                    X_NESS[word] = tag
                elif word[-3:] == 'ion':
                    X_ION[word] = tag
                elif word[-2:] == 'or':
                    X_OR[word] = tag
                elif word[-3:] == 'ant':
                    X_ANT[word] = tag
                elif word[-4:] == 'ship':
                    X_SHIP[word] = tag
                elif word[-2:] == 'th':
                    X_TH[word] = tag
                elif word[-3:] == 'ity':
                    X_ITY[word] = tag
                elif word[-3:] == 'ary':
                    X_ARY[word] = tag
                elif word[-3:] == 'ing':
                    X_ING[word] = tag
                elif word[-2:] == 'ly':
                    X_LY[word] = tag
                elif word[-2:] == 'ed':
                    X_ED[word] = tag
                elif word[-1:] == 's':
                    X_S[word] = tag
                elif word[-2:] == 'er':
                    X_ER[word] = tag
                else:
                    hapax[word] = tag

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
            if word in X_EST:
                em_probs[tag][word] = math.log(laplace * est_probs[tag] + em_probs[tag][word]) - math.log(tag_map[tag] + laplace * est_probs[tag] * (1 + tag_word_map[tag]))
            elif word in X_FUL:
                em_probs[tag][word] = math.log(laplace * ful_probs[tag] + em_probs[tag][word]) - math.log(tag_map[tag] + laplace * ful_probs[tag] * (1 + tag_word_map[tag]))
            elif word in X_ABLE:
                em_probs[tag][word] = math.log(laplace * able_probs[tag] + em_probs[tag][word]) - math.log(tag_map[tag] + laplace * able_probs[tag] * (1 + tag_word_map[tag]))    
            elif word in X_EN:
                em_probs[tag][word] = math.log(laplace * en_probs[tag] + em_probs[tag][word]) - math.log(tag_map[tag] + laplace * en_probs[tag] * (1 + tag_word_map[tag]))
            elif word in X_ISE:
                em_probs[tag][word] = math.log(laplace * ise_probs[tag] + em_probs[tag][word]) - math.log(tag_map[tag] + laplace * ise_probs[tag] * (1 + tag_word_map[tag]))
            elif word in X_IVE:
                em_probs[tag][word] = math.log(laplace * ive_probs[tag] + em_probs[tag][word]) - math.log(tag_map[tag] + laplace * ive_probs[tag] * (1 + tag_word_map[tag]))
            elif word in X_LESS:
                em_probs[tag][word] = math.log(laplace * less_probs[tag] + em_probs[tag][word]) - math.log(tag_map[tag] + laplace * less_probs[tag] * (1 + tag_word_map[tag]))
            elif word in X_NESS:
                em_probs[tag][word] = math.log(laplace * ness_probs[tag] + em_probs[tag][word]) - math.log(tag_map[tag] + laplace * ness_probs[tag] * (1 + tag_word_map[tag]))
            elif word in X_ION:
                em_probs[tag][word] = math.log(laplace * ion_probs[tag] + em_probs[tag][word]) - math.log(tag_map[tag] + laplace * ion_probs[tag] * (1 + tag_word_map[tag]))
            elif word in X_OR:
                em_probs[tag][word] = math.log(laplace * or_probs[tag] + em_probs[tag][word]) - math.log(tag_map[tag] + laplace * or_probs[tag] * (1 + tag_word_map[tag]))
            elif word in X_ANT:
                em_probs[tag][word] = math.log(laplace * ant_probs[tag] + em_probs[tag][word]) - math.log(tag_map[tag] + laplace * ant_probs[tag] * (1 + tag_word_map[tag]))
            elif word in X_SHIP:
                em_probs[tag][word] = math.log(laplace * ship_probs[tag] + em_probs[tag][word]) - math.log(tag_map[tag] + laplace * ship_probs[tag] * (1 + tag_word_map[tag]))
            elif word in X_TH:
                em_probs[tag][word] = math.log(laplace * th_probs[tag] + em_probs[tag][word]) - math.log(tag_map[tag] + laplace * th_probs[tag] * (1 + tag_word_map[tag]))
            elif word in X_ITY:
                em_probs[tag][word] = math.log(laplace * ity_probs[tag] + em_probs[tag][word]) - math.log(tag_map[tag] + laplace * ity_probs[tag] * (1 + tag_word_map[tag]))
            elif word in X_ARY:
                em_probs[tag][word] = math.log(laplace * ary_probs[tag] + em_probs[tag][word]) - math.log(tag_map[tag] + laplace * ary_probs[tag] * (1 + tag_word_map[tag]))
            elif word in X_ING:
                em_probs[tag][word] = math.log(laplace * ing_probs[tag] + em_probs[tag][word]) - math.log(tag_map[tag] + laplace * ing_probs[tag] * (1 + tag_word_map[tag]))
            elif word in X_LY:
                em_probs[tag][word] = math.log(laplace * ly_probs[tag] + em_probs[tag][word]) - math.log(tag_map[tag] + laplace * ly_probs[tag] * (1 + tag_word_map[tag]))
            elif word in X_ED:
                em_probs[tag][word] = math.log(laplace * ed_probs[tag] + em_probs[tag][word]) - math.log(tag_map[tag] + laplace * ed_probs[tag] * (1 + tag_word_map[tag]))
            elif word in X_S:
                em_probs[tag][word] = math.log(laplace * s_probs[tag] + em_probs[tag][word]) - math.log(tag_map[tag] + laplace * s_probs[tag] * (1 + tag_word_map[tag]))
            elif word in X_ER:
                em_probs[tag][word] = math.log(laplace * er_probs[tag] + em_probs[tag][word]) - math.log(tag_map[tag] + laplace * er_probs[tag] * (1 + tag_word_map[tag]))
            elif word in hapax:
                em_probs[tag][word] = math.log(laplace * hapax_probs[tag] + em_probs[tag][word]) - math.log(tag_map[tag] + laplace * hapax_probs[tag] * (1 + tag_word_map[tag]))
            else:
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
            trans_probs[tag1][tag2] = math.log(laplace + trans_probs[tag1][tag2]) - math.log(tag_map[tag1] + laplace * (1 + len(tag_word_map)))
    return trans_probs

def viterbi_3(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    out = []
    laplace = 1e-10
    tag_map, word_map = make_map(train)
    suffix_probs = count_hapax(train, tag_map, word_map, laplace)

    est_probs = suffix_probs[0]
    ful_probs = suffix_probs[1]
    able_probs = suffix_probs[2]
    en_probs = suffix_probs[3]
    ise_probs = suffix_probs[4]
    ive_probs = suffix_probs[5]
    less_probs = suffix_probs[6] 
    ness_probs = suffix_probs[7]
    ion_probs = suffix_probs[8]
    or_probs = suffix_probs[9] 
    ant_probs = suffix_probs[10]
    ship_probs = suffix_probs[11]
    th_probs = suffix_probs[12]
    ity_probs = suffix_probs[13]
    ary_probs = suffix_probs[14]
    ing_probs = suffix_probs[15]
    ly_probs = suffix_probs[16]
    ed_probs = suffix_probs[17]
    s_probs = suffix_probs[18]
    er_probs = suffix_probs[19]
    hapax_probs = suffix_probs[20]

    ini = {}
    for tag in tag_map:
        ini[tag] = 0
    for sentence in train:
        ini[sentence[1][1]] += 1
    for tag in tag_map:
        ini[tag] = math.log(ini[tag] + laplace) - math.log(len(train) + laplace * (1 + len(tag_map)))
    em_probs, tag_word_map = get_em_probs(train, tag_map, word_map, laplace, suffix_probs)
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
                if sentence[1][-3:] == 'est':
                    p1 = math.log(laplace * est_probs[tag]) - math.log(tag_map[tag] + laplace * est_probs[tag] * (1 + tag_word_map[tag]))
                elif sentence[1][-3:] == 'ful':
                    p1 = math.log(laplace * ful_probs[tag]) - math.log(tag_map[tag] + laplace * ful_probs[tag] * (1 + tag_word_map[tag]))
                elif sentence[1][-3:] == 'able':
                    p1 = math.log(laplace * able_probs[tag]) - math.log(tag_map[tag] + laplace * able_probs[tag] * (1 + tag_word_map[tag]))
                elif sentence[1][-3:] == 'en':
                    p1 = math.log(laplace * en_probs[tag]) - math.log(tag_map[tag] + laplace * en_probs[tag] * (1 + tag_word_map[tag]))
                elif sentence[1][-3:] == 'ise':
                    p1 = math.log(laplace * ise_probs[tag]) - math.log(tag_map[tag] + laplace * ise_probs[tag] * (1 + tag_word_map[tag]))
                elif sentence[1][-3:] == 'ive':
                    p1 = math.log(laplace * ive_probs[tag]) - math.log(tag_map[tag] + laplace * ive_probs[tag] * (1 + tag_word_map[tag]))
                elif sentence[1][-3:] == 'less':
                    p1 = math.log(laplace * less_probs[tag]) - math.log(tag_map[tag] + laplace * less_probs[tag] * (1 + tag_word_map[tag]))
                elif sentence[1][-3:] == 'ness':
                    p1 = math.log(laplace * ness_probs[tag]) - math.log(tag_map[tag] + laplace * ness_probs[tag] * (1 + tag_word_map[tag]))
                elif sentence[1][-3:] == 'ion':
                    p1 = math.log(laplace * ion_probs[tag]) - math.log(tag_map[tag] + laplace * ion_probs[tag] * (1 + tag_word_map[tag]))
                elif sentence[1][-3:] == 'or':
                    p1 = math.log(laplace * or_probs[tag]) - math.log(tag_map[tag] + laplace * or_probs[tag] * (1 + tag_word_map[tag]))
                elif sentence[1][-3:] == 'ant':
                    p1 = math.log(laplace * ant_probs[tag]) - math.log(tag_map[tag] + laplace * ant_probs[tag] * (1 + tag_word_map[tag]))
                elif sentence[1][-4:] == 'ship':
                    p1 = math.log(laplace * ship_probs[tag]) - math.log(tag_map[tag] + laplace * ship_probs[tag] * (1 + tag_word_map[tag]))
                elif sentence[1][-3:] == 'th':
                    p1 = math.log(laplace * th_probs[tag]) - math.log(tag_map[tag] + laplace * th_probs[tag] * (1 + tag_word_map[tag]))
                elif sentence[1][-3:] == 'ity':
                    p1 = math.log(laplace * ity_probs[tag]) - math.log(tag_map[tag] + laplace * ity_probs[tag] * (1 + tag_word_map[tag]))
                elif sentence[1][-3:] == 'ary':
                    p1 = math.log(laplace * ary_probs[tag]) - math.log(tag_map[tag] + laplace * ary_probs[tag] * (1 + tag_word_map[tag]))
                elif sentence[1][-3:] == 'ing':
                    p1 = math.log(laplace * ing_probs[tag]) - math.log(tag_map[tag] + laplace * ing_probs[tag] * (1 + tag_word_map[tag]))
                elif sentence[1][-2:] == 'ly':
                    p1 = math.log(laplace * ly_probs[tag]) - math.log(tag_map[tag] + laplace * ly_probs[tag] * (1 + tag_word_map[tag]))
                elif sentence[1][-2:] == 'ed':
                    p1 = math.log(laplace * ed_probs[tag]) - math.log(tag_map[tag] + laplace * ed_probs[tag] * (1 + tag_word_map[tag]))
                elif sentence[1][-1:] == 's':
                    p1 = math.log(laplace * s_probs[tag]) - math.log(tag_map[tag] + laplace * s_probs[tag] * (1 + tag_word_map[tag]))
                elif sentence[1][-2:] == 'er':
                    p1 = math.log(laplace * er_probs[tag]) - math.log(tag_map[tag] + laplace * er_probs[tag] * (1 + tag_word_map[tag]))
                else:
                    p1 = math.log(laplace * hapax_probs[tag]) - math.log(tag_map[tag] + laplace * hapax_probs[tag] * (1 + tag_word_map[tag]))
            trellis[tag][1] = ini[tag] + p1
            back[tag][1] = 0

        for i in range(2, n):
            for tag in tag_map:
                if sentence[i] in em_probs[tag]:
                    p = em_probs[tag][sentence[i]]
                else:
                    if sentence[i][-3:] == 'est':
                        p = math.log(laplace * est_probs[tag]) - math.log(tag_map[tag] + laplace * est_probs[tag] * (1 + tag_word_map[tag]))
                    elif sentence[i][-3:] == 'ful':
                        p = math.log(laplace * ful_probs[tag]) - math.log(tag_map[tag] + laplace * ful_probs[tag] * (1 + tag_word_map[tag]))
                    elif sentence[i][-3:] == 'able':
                        p = math.log(laplace * able_probs[tag]) - math.log(tag_map[tag] + laplace * able_probs[tag] * (1 + tag_word_map[tag]))
                    elif sentence[i][-3:] == 'en':
                        p = math.log(laplace * en_probs[tag]) - math.log(tag_map[tag] + laplace * en_probs[tag] * (1 + tag_word_map[tag]))
                    elif sentence[i][-3:] == 'ise':
                        p = math.log(laplace * ise_probs[tag]) - math.log(tag_map[tag] + laplace * ise_probs[tag] * (1 + tag_word_map[tag]))
                    elif sentence[i][-3:] == 'ive':
                        p = math.log(laplace * ive_probs[tag]) - math.log(tag_map[tag] + laplace * ive_probs[tag] * (1 + tag_word_map[tag]))
                    elif sentence[i][-3:] == 'less':
                        p = math.log(laplace * less_probs[tag]) - math.log(tag_map[tag] + laplace * less_probs[tag] * (1 + tag_word_map[tag]))
                    elif sentence[i][-3:] == 'ness':
                        p = math.log(laplace * ness_probs[tag]) - math.log(tag_map[tag] + laplace * ness_probs[tag] * (1 + tag_word_map[tag]))
                    elif sentence[i][-3:] == 'ion':
                        p = math.log(laplace * ion_probs[tag]) - math.log(tag_map[tag] + laplace * ion_probs[tag] * (1 + tag_word_map[tag]))
                    elif sentence[i][-3:] == 'or':
                        p = math.log(laplace * or_probs[tag]) - math.log(tag_map[tag] + laplace * or_probs[tag] * (1 + tag_word_map[tag]))
                    elif sentence[i][-3:] == 'ant':
                        p = math.log(laplace * ant_probs[tag]) - math.log(tag_map[tag] + laplace * ant_probs[tag] * (1 + tag_word_map[tag]))
                    elif sentence[i][-1:] == 'ship':
                        p = math.log(laplace * ship_probs[tag]) - math.log(tag_map[tag] + laplace * ship_probs[tag] * (1 + tag_word_map[tag]))
                    elif sentence[i][-3:] == 'th':
                        p = math.log(laplace * th_probs[tag]) - math.log(tag_map[tag] + laplace * th_probs[tag] * (1 + tag_word_map[tag]))
                    elif sentence[i][-3:] == 'ity':
                        p = math.log(laplace * ity_probs[tag]) - math.log(tag_map[tag] + laplace * ity_probs[tag] * (1 + tag_word_map[tag]))
                    elif sentence[i][-3:] == 'ary':
                        p = math.log(laplace * ary_probs[tag]) - math.log(tag_map[tag] + laplace * ary_probs[tag] * (1 + tag_word_map[tag]))
                    elif sentence[i][-3:] == 'ing':
                        p = math.log(laplace * ing_probs[tag]) - math.log(tag_map[tag] + laplace * ing_probs[tag] * (1 + tag_word_map[tag]))
                    elif sentence[i][-2:] == 'ly':
                        p = math.log(laplace * ly_probs[tag]) - math.log(tag_map[tag] + laplace * ly_probs[tag] * (1 + tag_word_map[tag]))
                    elif sentence[i][-2:] == 'ed':
                        p = math.log(laplace * ed_probs[tag]) - math.log(tag_map[tag] + laplace * ed_probs[tag] * (1 + tag_word_map[tag]))
                    elif sentence[i][-1:] == 's':
                        p = math.log(laplace * s_probs[tag]) - math.log(tag_map[tag] + laplace * s_probs[tag] * (1 + tag_word_map[tag]))
                    elif sentence[i][-2:] == 'er':
                        p = math.log(laplace * er_probs[tag]) - math.log(tag_map[tag] + laplace * er_probs[tag] * (1 + tag_word_map[tag]))
                    else:
                        p = math.log(laplace * hapax_probs[tag]) - math.log(tag_map[tag] + laplace * hapax_probs[tag] * (1 + tag_word_map[tag]))
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