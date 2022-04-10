# -*- coding: utf-8 -*-

'''
   Read data from JSON files,
   in the meantime, we do preprocess like capitalize the first character of a sentence or normalize digits
'''
import os

import json
from collections import Counter
# from stanfordcorenlp import StanfordCoreNLP
import numpy as np
import argparse
import pandas as pd

from io_utils import read_yaml, read_lines, read_json_lines, load_embedding_dict, save_pickle, read_pickle
from str_utils import capitalize_first_char, normalize_tok, normalize_sent, collapse_role_type
from vocab import Vocab
from actions import Actions
from sklearn.model_selection import train_test_split
from nltk.parse import CoreNLPDependencyParser
from tqdm import tqdm

joint_config = read_yaml('joint_config.yaml')

parser = argparse.ArgumentParser(description='this is a description')
parser.add_argument('--seed', '-s', required=False, type=int, default=joint_config['random_seed'])
args = parser.parse_args()
joint_config['random_seed'] = args.seed
print('seed:', joint_config['random_seed'])

np.random.seed(joint_config['random_seed'])

data_config = read_yaml('data_config.yaml')

data_dir = data_config['data_dir']
cur_dataset_dir = data_config['cur_dataset_dir']
embedding_dir = data_config['embedding_dir']
embedding_file = data_config['embedding_file']
embedding_type = data_config['embedding_type']

normalize_digits = data_config['normalize_digits']
lower_case = data_config['lower_case']

vocab_dir = data_config['vocab_dir']
token_vocab_file = os.path.join(vocab_dir, data_config['token_vocab_file'])
char_vocab_file = os.path.join(vocab_dir, data_config['char_vocab_file'])
action_vocab_file = os.path.join(vocab_dir, data_config['action_vocab_file'])
polarity_vocab_file = os.path.join(vocab_dir, data_config['polarity_vocab_file'])
# prd_type_vocab_file = os.path.join(vocab_dir, data_config['prd_type_vocab_file'])
# role_type_vocab_file = os.path.join(vocab_dir, data_config['role_type_vocab_file'])
pos_vocab_file = os.path.join(vocab_dir, data_config['pos_vocab_file'])
dep_type_vocab_file = os.path.join(vocab_dir, data_config['dep_type_vocab_file'])

embedd_dict, embedd_dim = None, None

lap_14_train_txt = os.path.join(data_config['lap_14'], 'train.txt')
lap_14_train_pair = os.path.join(data_config['lap_14'], '14lap_pair/train_pair.pkl')
lap_14_dev_txt = os.path.join(data_config['lap_14'], 'dev.txt')
lap_14_dev_pair = os.path.join(data_config['lap_14'], '14lap_pair/dev_pair.pkl')
lap_14_test_txt = os.path.join(data_config['lap_14'], 'test.txt')
lap_14_test_pair = os.path.join(data_config['lap_14'], '14lap_pair/test_pair.pkl')

res_14_train_txt = os.path.join(data_config['res_14'], 'train.txt')
res_14_train_pair = os.path.join(data_config['res_14'], '14res_pair/train_pair.pkl')
res_14_dev_txt = os.path.join(data_config['res_14'], 'dev.txt')
res_14_dev_pair = os.path.join(data_config['res_14'], '14res_pair/dev_pair.pkl')
res_14_test_txt = os.path.join(data_config['res_14'], 'test.txt')
res_14_test_pair = os.path.join(data_config['res_14'], '14res_pair/test_pair.pkl')

res_15_train_txt = os.path.join(data_config['res_15'], 'train.txt')
res_15_train_pair = os.path.join(data_config['res_15'], '15res_pair/train_pair.pkl')
res_15_dev_txt = os.path.join(data_config['res_15'], 'dev.txt')
res_15_dev_pair = os.path.join(data_config['res_15'], '15res_pair/dev_pair.pkl')
res_15_test_txt = os.path.join(data_config['res_15'], 'test.txt')
res_15_test_pair = os.path.join(data_config['res_15'], '15res_pair/test_pair.pkl')

res_16_train_txt = os.path.join(data_config['res_16'], 'train.txt')
res_16_train_pair = os.path.join(data_config['res_16'], '16res_pair/train_pair.pkl')
res_16_dev_txt = os.path.join(data_config['res_16'], 'dev.txt')
res_16_dev_pair = os.path.join(data_config['res_16'], '16res_pair/dev_pair.pkl')
res_16_test_txt = os.path.join(data_config['res_16'], 'test.txt')
res_16_test_pair = os.path.join(data_config['res_16'], '16res_pair/test_pair.pkl')

POLARITY_DICT = {'NEU': 0, 'POS': 1, 'NEG': 2}
POLARITY_DICT_REV = {v: k for k, v in POLARITY_DICT.items()}


def load_old_triplet(txt_path, pair_path):
    """

    :param txt_path: the original annotation file path
    :param pair_path: the processed pair file path
    :return:
    """
    pairs = read_pickle(pair_path)
    data_list = []
    # texts = read_lines(txt_path, encoding='utf-8', return_list=True)
    with open(txt_path, encoding='utf-8') as f:
        texts = f.readlines()
    assert len(pairs) == len(texts)
    for idx, (t, p) in enumerate(zip(texts, pairs)):
        data_dic = {}
        temp = t.split('####')
        words = temp[0].split(' ')
        # pos_tag = st.pos_tag(temp[0])
        # dep = st.dependency_parse(temp[0])
        opinion = []
        opinion_idx = []
        aspect = []
        aspect_idx = []
        polarity = []
        ps = []
        for i in p:
            a = words[i[0][0]: i[0][-1]+1] if len(i[0]) > 1 else [words[i[0][0]]]
            o = words[i[1][0]: i[1][-1]+1] if len(i[1]) > 1 else [words[i[1][0]]]
            ps.append((a, o, POLARITY_DICT_REV[i[2]]))
            if i[0] not in aspect:
                aspect.append(a)
                aspect_idx.append(i[0])
            if i[1] not in opinion:
                opinion.append(o)
                opinion_idx.append(i[1])

        data_dic['words'] = words
        data_dic['aspects'] = aspect
        data_dic['aspects_idx'] = aspect_idx
        data_dic['opinions'] = opinion
        data_dic['opinions_idx'] = opinion_idx
        data_dic['pair'] = ps
        data_dic['pair_idx'] = p
        data_list.append(data_dic)

    return data_list


def get_dep(token_list, depparser):
    res = []
    parser_res = depparser.parse(token_list)
    for i in parser_res:
        temp = i.to_conll(4).strip().split('\n')
        for t in temp:
            res.append(t.split('\t'))
    return res


def get_new_ides(new_tokens, ori_tokens, ori_oht_token_list, ori_oht_ides_list, depparser):
    new_len = len(new_tokens)
    ori_len = len(ori_tokens)
    chazhi = new_len - ori_len
    if ori_oht_token_list[0] == 'doesnt':
        tokenized_tokens = ['does', 'nt']
    else:
        tokenized_tokens = list(depparser.tokenize(' '.join([normalize_tok(w) for w in ori_oht_token_list])))
    try:
        new_ht_s = new_tokens.index(tokenized_tokens[0], ori_oht_ides_list[0], ori_oht_ides_list[0] + chazhi + 1)
    except ValueError as ve:
        print('index start error: ', ve)
        new_ht_s = new_tokens.index(''.join(tokenized_tokens[:2]), ori_oht_ides_list[0], ori_oht_ides_list[0] + chazhi + 1)
        print(''.join(tokenized_tokens[:2]), ' index correct.')

    try:
        new_ht_e = max(new_tokens.index(tokenized_tokens[-1], ori_oht_ides_list[-1], ori_oht_ides_list[-1] + chazhi + 1),
                       new_ht_s + len(tokenized_tokens) - 1)
    except ValueError as ve:
        print('index end error: ', ve)
        new_ht_e = max(new_tokens.index(''.join(tokenized_tokens[-2:]), ori_oht_ides_list[-1], ori_oht_ides_list[-1] + chazhi + 1),
                       new_ht_s + len(tokenized_tokens) - 2)
        print(''.join(tokenized_tokens[-2:]), ' index correct.')

    temp = [x for x in range(max(new_ht_s, new_ht_e - len(tokenized_tokens) + 1), new_ht_e + 1)]
    return temp


def load_triplet_data(txt_path):
    """

        :param txt_path: the original annotation file path
        :return:
        """
    data_list = []
    with open(txt_path, encoding='utf-8') as f:
        texts = f.readlines()
    for idx, t in enumerate(texts):
        data_dic = {}
        temp = t.split('####')
        words = temp[0].split(' ')
        p = eval(temp[3])
        ori_words = [normalize_tok(w) for w in words]
        new_words = []
        temp = get_dep(ori_words, depparser)
        pos_list = []
        dep_rel_list = []
        dep_head_list = []
        for t in temp:
            new_words.append(t[0])
            pos_list.append(t[1])
            dep_head_list.append(int(t[2])-1)
            dep_rel_list.append(t[3])

        opinion = []
        opinion_idx = []
        aspect = []
        aspect_idx = []
        polarity = []
        pair = []
        new_pair_idx = []
        for idx, i in enumerate(p):
            a = words[i[0][0]: i[0][-1] + 1] if len(i[0]) > 1 else [words[i[0][0]]]
            new_a_idx = get_new_ides(new_words, ori_words, a, i[0], depparser) if len(new_words) != len(
                ori_words) else i[0]
            new_a = new_words[new_a_idx[0]: new_a_idx[-1] + 1] if len(new_words) != len(ori_words) else a

            o = words[i[1][0]: i[1][-1] + 1] if len(i[1]) > 1 else [words[i[1][0]]]
            new_o_idx = get_new_ides(new_words, ori_words, o, i[1], depparser) if len(new_words) != len(
                ori_words) else i[1]
            new_o = new_words[new_o_idx[0]: new_o_idx[-1] + 1] if len(new_words) != len(ori_words) else o

            new_pair_idx.append((new_a_idx, new_o_idx, i[2]))
            pair.append((new_a, new_o, i[2]))

            if new_a_idx not in aspect_idx:
                aspect.append(new_a)
                aspect_idx.append(new_a_idx)
            if new_o_idx not in opinion_idx:
                opinion.append(new_o)
                opinion_idx.append(new_o_idx)
            polarity.append(i[2])

        data_dic['words'] = new_words
        data_dic['pos'] = pos_list
        data_dic['dep_head'] = dep_head_list
        data_dic['dep_rel'] = dep_rel_list
        data_dic['pos'] = pos_list
        data_dic['aspects'] = aspect
        data_dic['aspects_idx'] = aspect_idx
        data_dic['opinions'] = opinion
        data_dic['opinions_idx'] = opinion_idx
        data_dic['pair'] = pair
        data_dic['pair_idx'] = new_pair_idx
        data_dic['polarity'] = polarity
        data_list.append(data_dic)

    return data_list


def read_embedding():
    global embedd_dict, embedd_dim
    embedd_dict, embedd_dim = load_embedding_dict(embedding_type,
                                                  os.path.join(embedding_dir, embedding_file),
                                                  normalize_digits=normalize_digits)
    print('Embedding type %s, file %s' % (embedding_type, embedding_file))


def build_vocab():
    token_list = []
    char_list = []
    polarity_list = []

    aspects_list = []
    opinions_list = []
    actions_list = []
    dep_type_list = []
    pos_type_list = []

    for inst in tqdm(train_list, total=len(train_list)):
        words = inst['words']
        aspects = inst['aspects_idx']  # idx, prds_type
        opinions = inst['opinions_idx']  # arg_id, prd_id, role
        pair_idx = inst['pair_idx']
        polarity = inst['polarity']
        dep_type_list.extend(inst['dep_rel'])
        pos_type_list.extend(inst['pos'])

        for word in words:
            word = normalize_tok(word, lower_case, normalize_digits)
            if embedd_dict is not None and (word in embedd_dict or word.lower() in embedd_dict):
                token_list.append(word)
                char_list.extend(list(word))

        aspects_list.extend(aspects)
        opinions_list.extend(opinions)
        polarity_list.extend(polarity)

        actions = Actions.make_oracle(words, pair_idx, aspects, opinions)
        actions_list.extend(actions)

    train_token_set = set(token_list)

    for inst in tqdm(dev_list, total=len(dev_list)):
        words = inst['words']
        aspects = inst['aspects_idx']  # idx, prds_type
        opinions = inst['opinions_idx']  # arg_id, prd_id, role
        pair_idx = inst['pair_idx']
        polarity = inst['polarity']
        dep_type_list.extend(inst['dep_rel'])
        pos_type_list.extend(inst['pos'])

        for word in words:
            word = normalize_tok(word, lower_case, normalize_digits)
            if embedd_dict is not None and (word in embedd_dict or word.lower() in embedd_dict):
                token_list.append(word)
                char_list.extend(list(word))

        aspects_list.extend(aspects)
        opinions_list.extend(opinions)
        polarity_list.extend(polarity)

        actions = Actions.make_oracle(words, pair_idx, aspects, opinions)
        actions_list.extend(actions)

    # test_oo_train_but_in_glove = 0
    for inst in tqdm(test_list, total=len(test_list)):
        words = inst['words']
        aspects = inst['aspects_idx']  # idx, prds_type
        opinions = inst['opinions_idx']  # arg_id, prd_id, role
        pair_idx = inst['pair_idx']
        polarity = inst['polarity']
        dep_type_list.extend(inst['dep_rel'])
        pos_type_list.extend(inst['pos'])

        for word in words:
            word = normalize_tok(word, lower_case, normalize_digits)
            if embedd_dict is not None and (word in embedd_dict or word.lower() in embedd_dict):
                token_list.append(word)
                char_list.extend(list(word))

        aspects_list.extend(aspects)
        opinions_list.extend(opinions)
        polarity_list.extend(polarity)

        actions = Actions.make_oracle(words, pair_idx, aspects, opinions)
        actions_list.extend(actions)

    print('--------token_vocab---------------')
    token_vocab = Vocab()
    token_vocab.add_spec_toks(unk_tok=True, pad_tok=False)
    token_vocab.add_counter(Counter(token_list))
    token_vocab.save(token_vocab_file)
    print(token_vocab)

    print('--------char_vocab---------------')
    char_vocab = Vocab()
    char_vocab.add_spec_toks(unk_tok=True, pad_tok=False)
    char_vocab.add_counter(Counter(char_list))
    char_vocab.save(char_vocab_file)
    print(char_vocab)

    print('--------action_vocab---------------')
    action_vocab = Vocab()
    action_vocab.add_spec_toks(pad_tok=False, unk_tok=False)
    action_vocab.add_counter(Counter(actions_list))
    action_vocab.save(action_vocab_file)
    print(action_vocab)

    print('--------polarity_vocab---------------')
    polarity_vocab = Vocab()
    polarity_vocab.add_spec_toks(pad_tok=False, unk_tok=False, null_tok=True)
    polarity_vocab.add_counter(Counter(polarity_list))
    polarity_vocab.save(polarity_vocab_file)
    print(polarity_vocab)

    print('--------pos_vocab---------------')
    pos_vocab = Vocab()
    pos_vocab.add_spec_toks(pad_tok=False, unk_tok=False, null_tok=True)
    pos_vocab.add_counter(Counter(pos_type_list))
    pos_vocab.save(pos_vocab_file)
    print(pos_vocab)

    print('--------dep_type_vocab---------------')
    dep_type_vocab = Vocab()
    dep_type_vocab.add_spec_toks(pad_tok=False, unk_tok=False, null_tok=True)
    dep_type_vocab.add_counter(Counter(dep_type_list))
    dep_type_vocab.save(dep_type_vocab_file)
    print(dep_type_vocab)


def construct_instance(inst_list, token_vocab, char_vocab, action_vocab, polarity_vocab,
                       pos_vocab=None, dep_type_vocab=None, is_train=True):
    word_num = 0
    processed_inst_list = []
    for inst in inst_list:

        words = inst['words']
        aspects = inst['aspects_idx']
        opinions = inst['opinions_idx']
        pair = inst['pair_idx']
        pos = inst['pos']
        dep_head = inst['dep_head']
        dep_labels = inst['dep_rel']

        if is_train and len(pair) == 0:
            continue

        words_processed = []
        word_indices = []
        char_indices = []
        for word in words:
            word = normalize_tok(word, lower_case, normalize_digits)
            words_processed.append(word)
            word_idx = token_vocab.get_index(word)
            word_indices.append(word_idx)
            char_indices.append([char_vocab.get_index(c) for c in word])

        inst['words'] = words_processed
        inst['word_indices'] = word_indices
        inst['char_indices'] = char_indices

        inst['pos_indices'] = [pos_vocab.get_index(p) for p in pos]
        inst['dep_label_indices'] = [dep_type_vocab.get_index(p) for p in dep_labels]
        pair_idx = [(i[0], i[1], polarity_vocab.get_index(i[2])) for i in pair]
        inst['pair_idx'] = pair_idx
        actions = Actions.make_oracle(words, pair_idx, aspects, opinions)
        inst['actions'] = actions
        inst['action_indices'] = [action_vocab.get_index(act) for act in actions]

        inst['sent_range'] = list(range(word_num, word_num + len(words)))
        word_num += len(words)

        processed_inst_list.append(inst)

    return processed_inst_list


def pickle_data():
    token_vocab = Vocab.load(token_vocab_file)
    char_vocab = Vocab.load(char_vocab_file)
    action_vocab = Vocab.load(action_vocab_file)
    polarity_vocab = Vocab.load(polarity_vocab_file)
    pos_vocab = Vocab.load(pos_vocab_file)
    dep_type_vocab = Vocab.load(dep_type_vocab_file)

    processed_train = construct_instance(train_list, token_vocab, char_vocab, action_vocab, polarity_vocab, pos_vocab, dep_type_vocab)
    processed_dev = construct_instance(dev_list, token_vocab, char_vocab, action_vocab, polarity_vocab, pos_vocab, dep_type_vocab, False)
    processed_test = construct_instance(test_list, token_vocab, char_vocab, action_vocab, polarity_vocab, pos_vocab, dep_type_vocab, False)

    print('Saving pickle to ', inst_pl_file)
    print('Saving sent size Train: %d, Dev: %d, Test:%d' % (
        len(processed_train), len(processed_dev), len(processed_test)))
    save_pickle(inst_pl_file, [processed_train, processed_dev, processed_test, token_vocab, char_vocab,  action_vocab, polarity_vocab, pos_vocab, dep_type_vocab])

    scale = np.sqrt(3.0 / embedd_dim)
    vocab_dict = token_vocab.tok2idx
    table = np.empty([len(vocab_dict), embedd_dim], dtype=np.float32)
    oov = 0
    for word, index in tqdm(vocab_dict.items(), total=vocab_dict.__len__()):
        if word in embedd_dict:
            embedding = embedd_dict[word]
        elif word.lower() in embedd_dict:
            embedding = embedd_dict[word.lower()]
        else:
            embedding = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
            oov += 1
        table[index, :] = embedding

    np.save(vec_npy_file, table)
    print('pretrained embedding oov: %d' % oov)
    print()


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


def count_instance_info(inst_lsit):
    a_term_num = 0
    a_term_len = []
    o_term_num = 0
    o_term_len = []
    a_o_term_dist = []
    overlap_num = 0
    pair_num = 0

    overlap_type_num_1 = {}

    for inst in inst_lsit:
        aspects = inst['aspects_idx']
        opinions = inst['opinions_idx']
        pair = inst['pair_idx']

        a_term_num += len(aspects)
        for idx, a in enumerate(aspects):
            a_term_len.append(len(a))

        o_term_num += len(opinions)
        for idx, o in enumerate(opinions):
            o_term_len.append(len(o))

        pair_num += len(pair)
        for p in pair:
            t1, t2 = p[0], p[1]
            if (t1[-1] > t2[0] and t1[0] < t2[0]) or (t2[-1] > t1[0] and t2[0] < t1[0]):
                a_o_term_dist.append(0)
            elif t1[-1] < t2[0]:
                a_o_term_dist.append(t2[0]-t1[-1]-1)
            elif t2[-1] < t1[0]:
                a_o_term_dist.append(t1[0]-t2[-1]-1)

        for k, v in inst['overlap'].items():
            if k in overlap_type_num_1.keys():
                overlap_type_num_1[k] += len(v)
            else:
                overlap_type_num_1[k] = len(v)

        overlap_num += inst['overlap_num']

    assert len(a_term_len) == a_term_num
    assert len(o_term_len) == o_term_num

    return a_term_num, a_term_len, o_term_num, o_term_len, a_o_term_dist, overlap_num, pair_num, overlap_type_num_1


def count_info(file_path, target_path=None):
    """
    avg. a/o term length,
    avg. triplet overlap number,
    avg. a/o term间距
    :param file_path:
    :return:
    """
    train, dev, test, _, _, _, _, _, _ = read_pickle(file_path)
    a_term_num = 0
    a_term_len = []
    o_term_num = 0
    o_term_len = []
    a_o_term_dist = []
    overlap_num = 0
    pair_num = 0
    f = open(target_path, 'a', encoding='utf-8')
    f.write('\n'+file_path.split('/')[-2]+'\n')
    t_a_term_num, t_a_term_len, t_o_term_num, t_o_term_len, t_a_o_term_dist, t_overlap_num, t_pair_num, overlap_type_num = count_instance_info(
        train)
    print("Train")
    print("\taspect num: {} \t opinion num : {} \t pair num: {} \t overlap num: {}".format(t_a_term_num, t_o_term_num, t_pair_num, t_overlap_num))
    f.writelines('Train\n')
    f.writelines("\taspect num: {} \t opinion num : {} \t pair num: {} \t overlap num: {}\n".format(t_a_term_num, t_o_term_num, t_pair_num, t_overlap_num))
    f.writelines("overlap type num: {}\n".format(overlap_type_num))
    # f.writelines("overlap type num: {}\n".format(overlap_type_num_1))
    a_term_num += t_a_term_num
    a_term_len.extend(t_a_term_len)
    o_term_num += t_o_term_num
    o_term_len.extend(t_o_term_len)
    a_o_term_dist.extend(t_a_o_term_dist)
    overlap_num += t_overlap_num
    pair_num += t_pair_num

    d_a_term_num, d_a_term_len, d_o_term_num, d_o_term_len, d_a_o_term_dist, d_overlap_num, d_pair_num, overlap_type_num = count_instance_info(
        dev)
    print("Dev")
    print("\taspect num: {} \t opinion num : {} \t pair num: {} \t overlap num: {}".format(d_a_term_num, d_o_term_num, d_pair_num, d_overlap_num))
    f.writelines("Dev\n")
    f.writelines("\taspect num: {} \t opinion num : {} \t pair num: {} \t overlap num: {}\n".format(d_a_term_num, d_o_term_num, d_pair_num, d_overlap_num))
    f.writelines("overlap type num: {}\n".format(overlap_type_num))
    # f.writelines("overlap type num: {}\n".format(overlap_type_num_1))
    a_term_num += d_a_term_num
    a_term_len.extend(d_a_term_len)
    o_term_num += d_o_term_num
    o_term_len.extend(d_o_term_len)
    a_o_term_dist.extend(d_a_o_term_dist)
    overlap_num += d_overlap_num
    pair_num += d_pair_num

    s_a_term_num, s_a_term_len, s_o_term_num, s_o_term_len, s_a_o_term_dist, s_overlap_num, s_pair_num, overlap_type_num = count_instance_info(
        test)
    print("Test")
    print("\taspect num: {} \t opinion num : {} \t pair num: {} \t overlap num: {}".format(s_a_term_num, s_o_term_num, s_pair_num, s_overlap_num))
    f.writelines("Test\n")
    f.writelines("\taspect num: {} \t opinion num : {} \t pair num: {} \t overlap num: {}\n".format(s_a_term_num, s_o_term_num, s_pair_num, s_overlap_num))
    f.writelines("overlap type num: {}\n".format(overlap_type_num))
    # f.writelines("overlap type num: {}\n".format(overlap_type_num_1))
    a_term_num += s_a_term_num
    a_term_len.extend(s_a_term_len)
    o_term_num += s_o_term_num
    o_term_len.extend(s_o_term_len)
    a_o_term_dist.extend(s_a_o_term_dist)
    overlap_num += s_overlap_num
    pair_num += s_pair_num

    print('Total')
    print("\taspect num: {} \t opinion num : {} \t pair num: {} \t overlap num: {}".format(a_term_num, o_term_num, pair_num, overlap_num))
    f.writelines('Total\n')
    f.write("\taspect num: {} \t opinion num : {} \t pair num: {} \t overlap num: {}\n".format(a_term_num, o_term_num, pair_num, overlap_num))

    a_term_len_all = sum(a_term_len)
    o_term_len_all = sum(o_term_len)
    a_o_term_dist_all = sum(a_o_term_dist)

    avg_a_term_len = a_term_len_all / a_term_num
    avg_o_term_len = o_term_len_all / o_term_num
    avg_term_dist = a_o_term_dist_all / pair_num
    print("avg_a_term_len: %.3f \t avg_o_term_len: %.3f \t avg_term_dist: %.3f" % (avg_a_term_len, avg_o_term_len, avg_term_dist))
    f.write("avg_a_term_len: %.3f \t avg_o_term_len: %.3f \t avg_term_dist: %.3f\n" % (avg_a_term_len, avg_o_term_len, avg_term_dist))
    f.close()


def count_14res_info(data):
    a_overlap_num, o_overlap_num = 0, 0
    train, dev, test, _, _, _, _ = read_pickle(data)
    train_a, train_o = 0, 0
    dev_a, dev_o = 0, 0
    test_a, test_o = 0, 0
    for inst in train:
        aspects = inst['aspects_idx']
        opinions = inst['opinions_idx']
        pair_idx = inst['pair_idx']

        a_term_dict = {}
        o_term_dict = {}
        for idx, a in enumerate(aspects):
            a_term_dict['a-' + str(idx)] = a

        for idx, o in enumerate(opinions):
            o_term_dict['o-' + str(idx)] = o
        pair_term = []
        for p in pair_idx:
            t1, t2 = p[0], p[1]
            pair_term.extend(get_key(a_term_dict, t1))
            pair_term.extend(get_key(o_term_dict, t2))
        res = Counter(pair_term)
        for p in pair_idx:
            a_key = get_key(a_term_dict, p[0])
            o_key = get_key(o_term_dict, p[1])
            c0 = res[a_key[0]]
            c1 = res[o_key[0]]
            if c0 > 1:
                train_a += 1
            elif c1 > 1:
                train_o += 1
    print('train: aspect overlap number {} opinion overlap number {}'.format(train_a, train_o))
    a_overlap_num += train_a
    o_overlap_num += train_o
    for inst in dev:
        aspects = inst['aspects_idx']
        opinions = inst['opinions_idx']
        pair_idx = inst['pair_idx']

        a_term_dict = {}
        o_term_dict = {}
        for idx, a in enumerate(aspects):
            a_term_dict['a-' + str(idx)] = a

        for idx, o in enumerate(opinions):
            o_term_dict['o-' + str(idx)] = o
        pair_term = []
        for p in pair_idx:
            t1, t2 = p[0], p[1]
            pair_term.extend(get_key(a_term_dict, t1))
            pair_term.extend(get_key(o_term_dict, t2))
        res = Counter(pair_term)
        for p in pair_idx:
            a_key = get_key(a_term_dict, p[0])
            o_key = get_key(o_term_dict, p[1])
            c0 = res[a_key[0]]
            c1 = res[o_key[0]]
            if c0 > 1:
                dev_a += 1
            elif c1 > 1:
                dev_o += 1
    print('dev: aspect overlap number {} opinion overlap number {}'.format(dev_a, dev_o))
    a_overlap_num += dev_a
    o_overlap_num += dev_o
    for inst in test:
        aspects = inst['aspects_idx']
        opinions = inst['opinions_idx']
        pair_idx = inst['pair_idx']

        a_term_dict = {}
        o_term_dict = {}
        for idx, a in enumerate(aspects):
            a_term_dict['a-' + str(idx)] = a

        for idx, o in enumerate(opinions):
            o_term_dict['o-' + str(idx)] = o
        pair_term = []
        for p in pair_idx:
            t1, t2 = p[0], p[1]
            pair_term.extend(get_key(a_term_dict, t1))
            pair_term.extend(get_key(o_term_dict, t2))
        res = Counter(pair_term)
        for p in pair_idx:
            a_key = get_key(a_term_dict, p[0])
            o_key = get_key(o_term_dict, p[1])
            c0 = res[a_key[0]]
            c1 = res[o_key[0]]
            if c0 > 1:
                test_a += 1
            elif c1 > 1:
                test_o += 1
    print('test: aspect overlap number {} opinion overlap number {}'.format(test_a, test_o))
    a_overlap_num += test_a
    o_overlap_num += test_o
    print('Total: aspect overlap number {} opinion overlap number {}'.format(a_overlap_num, o_overlap_num))


def is_nested_term(file_path):
    data = read_pickle(file_path)
    # train, dev, test = data
    print(file_path)
    for inst in data[0]:
        term = []
        term.extend(inst['aspects_idx'])
        term.extend(inst['opinions_idx'])
        for i in range(len(term)):
            for j in range(i+1, len(term)):
                if term[i][0] < term[j][0] < term[j][-1] < term[i][-1]:
                    print(inst['words'])
                    print(inst['pair_idx'])
                elif term[j][0] < term[i][0] < term[i][-1] < term[j][-1]:
                    print(inst['words'])
                    print(inst['pair_idx'])
                elif term[i][0] < term[j][0] < term[i][-1] < term[j][-1]:
                    print(inst['words'])
                    print(inst['pair_idx'])
                elif term[j][0] < term[i][0] < term[j][-1] < term[i][-1]:
                    print(inst['words'])
                    print(inst['pair_idx'])
                else:
                    continue

    print('xxx')


if __name__ == '__main__':

    train_list = []
    dev_list = []
    test_list = []

    depparser = CoreNLPDependencyParser(url='http://127.0.0.1:9000')

    lap_14_train = load_triplet_data(lap_14_train_txt)
    lap_14_dev = load_triplet_data(lap_14_dev_txt)
    lap_14_test = load_triplet_data(lap_14_test_txt)

    res_14_train = load_triplet_data(res_14_train_txt)
    res_14_dev = load_triplet_data(res_14_dev_txt)
    res_14_test = load_triplet_data(res_14_test_txt)

    res_15_train = load_triplet_data(res_15_train_txt)
    res_15_dev = load_triplet_data(res_15_dev_txt)
    res_15_test = load_triplet_data(res_15_test_txt)

    res_16_train = load_triplet_data(res_16_train_txt)
    res_16_dev = load_triplet_data(res_16_dev_txt)
    res_16_test = load_triplet_data(res_16_test_txt)

    print(
        '14_lap: {}, 14_res: {}, 15_res: {}, 16_res: {}'.format(len(lap_14_train), len(res_14_train), len(res_15_train),
                                                                len(res_16_train)))
    print(
        '14_lap: {}, 14_res: {}, 15_res: {}, 16_res: {}'.format(len(lap_14_dev), len(res_14_dev), len(res_15_dev),
                                                                len(res_16_dev)))
    print(
        '14_lap: {}, 14_res: {}, 15_res: {}, 16_res: {}'.format(len(lap_14_test), len(res_14_test), len(res_15_test),
                                                                len(res_16_test)))

    train_list.extend(lap_14_train)
    train_list.extend(res_14_train)
    train_list.extend(res_15_train)
    train_list.extend(res_16_train)

    dev_list.extend(lap_14_dev)
    dev_list.extend(res_14_dev)
    dev_list.extend(res_15_dev)
    dev_list.extend(res_16_dev)

    test_list.extend(lap_14_test)
    test_list.extend(res_14_test)
    test_list.extend(res_15_test)
    test_list.extend(res_16_test)

    read_embedding()
    build_vocab()

    pickle_dir = 'data/pickle/14lap'
    vec_npy_file = 'data/pickle/14lap/word_vec.npy'
    inst_pl_file = 'data/pickle/14lap/data.pl'
    train_list = lap_14_train
    dev_list = lap_14_dev
    test_list = lap_14_test
    pickle_data()

    pickle_dir = 'data/pickle/14res'
    vec_npy_file = 'data/pickle/14res/word_vec.npy'
    inst_pl_file = 'data/pickle/14res/data.pl'
    train_list = res_14_train
    dev_list = res_14_dev
    test_list = res_14_test
    pickle_data()

    pickle_dir = 'data/pickle/15res'
    vec_npy_file = 'data/pickle/15res/word_vec.npy'
    inst_pl_file = 'data/pickle/15res/data.pl'
    train_list = res_15_train
    dev_list = res_15_dev
    test_list = res_15_test
    pickle_data()
    #
    pickle_dir = 'data/pickle/16res'
    vec_npy_file = 'data/pickle/16res/word_vec.npy'
    inst_pl_file = 'data/pickle/16res/data.pl'
    train_list = res_16_train
    dev_list = res_16_dev
    test_list = res_16_test
    pickle_data()


    # file_path = 'data/pickle/14res/data.pl'
    # count_info(file_path, 'count_info_2.txt')
    # file_path = 'data/pickle/14res/data_2.pl'
    # count_info(file_path, 'count_info_2.txt')
    # file_path = 'data/pickle/15res/data_2.pl'
    # count_info(file_path, 'count_info_2.txt')
    # file_path = 'data/pickle/16res/data_2.pl'
    # count_info(file_path, 'count_info_2.txt')

    # count_14res_info('data/pickle/14res/data_2.pl')

    # is_nested_term(file_path)
    print('-_-!')
