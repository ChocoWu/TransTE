
from vocab import Vocab
from sklearn.metrics._classification import classification_report
import numpy as np
from collections import Counter


def to_set(input):
    out_set = set()
    out_type_set = set()
    for x in input:
        out_set.add(tuple(x[:-1]))
        out_type_set.add(tuple(x))

    return out_set, out_type_set


class AOPEval(object):

    def __init__(self, action_vocab, overlap_type=None):
        """

        :param action_vocab: dict
        :param overlap_type: List
        """
        self.action_vocab = action_vocab
        self.action_vocab.idx2tok = {v: k for k, v in action_vocab.tok2idx.items()}
        self.overlap_type = overlap_type
        self.reset()
        self.overlap_type_num_1 = {}

    def reset(self):
        self.correct_aspects = 0.
        self.correct_opinions = 0.
        self.num_pre_aspects = 0.
        self.num_gold_aspects = 0.
        self.num_pre_opinions = 0.
        self.num_gold_opinions = 0.

        self.num_pre_pair = 0.
        self.num_gold_pair = 0.
        self.correct_triplet = 0.
        self.correct_pair = 0.

        self.pred_ections = []
        self.gold_actions = []
        self.act_count = {x: {'act_str': self.action_vocab.idx2tok[x], 'gold': 0.0, 'pred': 0.0, 'correct': 0.0, 'p': 0.0, 'r': 0.0, 'f1': 0.0, 'transition': [0.0] * len(self.action_vocab)} for x in range(len(self.action_vocab))}
        self.overlap_count = {x: {'pred': 0.0, 'gold': 0.0, 'correct': 0.0, 'p': 0.0, 'r': 0.0, 'f1': 0.0} for x in self.overlap_type}

    def update(self, pred_aspects, gold_aspects,
               pred_opinions, gold_opinions,
               pred_pairs, gold_pairs, gold_overlap=None, eval_arg=True, words=None):

        def deleteDuplicatedElementFromList(x):
            c = []
            for i in x:
                if i not in c:
                    c.append(i)
            return c

        # pred_aspects = deleteDuplicatedElementFromList(pred_aspects)
        # gold_aspects = deleteDuplicatedElementFromList(gold_aspects)
        self.num_pre_aspects += len(pred_aspects)
        self.num_gold_aspects += len(gold_aspects)

        # pred_opinions = deleteDuplicatedElementFromList(pred_opinions)
        # gold_opinions = deleteDuplicatedElementFromList(gold_opinions)
        self.num_pre_opinions += len(pred_opinions)
        self.num_gold_opinions += len(gold_opinions)

        self.num_pre_pair += len(pred_pairs)
        self.num_gold_pair += len(gold_pairs)

        # a_p = {'a-' + str(x): a for x, a in enumerate(pred_aspects)}
        # a_g = {'a-' + str(x): a for x, a in enumerate(gold_aspects)}
        # o_p = {'o-' + str(x): a for x, a in enumerate(pred_opinions)}
        # o_g = {'o-' + str(x): a for x, a in enumerate(gold_opinions)}

        for i in gold_aspects:
            for j in pred_aspects:
                if i == j:
                    self.correct_aspects += 1
        for i in gold_opinions:
            for j in pred_opinions:
                if i == j:
                    self.correct_opinions += 1

        # gold_pairs_ = deleteDuplicatedElementFromList(gold_pairs)
        # pred_pairs_ = deleteDuplicatedElementFromList(pred_pairs)

        # for k, v in gold_overlap.items():
        #     if k in self.overlap_count.keys():
        #         self.overlap_count[k]['gold'] += len(v)
        #     else:
        #         self.overlap_count[k] = {'pred': 0.0, 'gold': 0.0, 'correct': 0.0, 'p': 0.0, 'r': 0.0, 'f1': 0.0}
        #         self.overlap_count[k]['gold'] = len(v)
        #
        # pred_overlap, pred_overlap_num = self.get_overlap(pred_pairs, a_p, o_p)
        # for k, v in pred_overlap.items():
        #     if k in self.overlap_count.keys():
        #         self.overlap_count[k]['pred'] += len(v)
        #     else:
        #         self.overlap_count[k] = {'pred': 0.0, 'gold': 0.0, 'correct': 0.0, 'p': 0.0, 'r': 0.0, 'f1': 0.0}
        #         self.overlap_count[k]['pred'] = len(v)

        # print("gold_overlap")
        # print(gold_overlap)
        # print("pred_overlap")
        # print(pred_overlap)
        # for k, p in pred_overlap.items():
        #     if k in gold_overlap.keys():
        #         g_p = gold_overlap[k]
        #         for i in p:
        #             for j in g_p:
        #                 if (i[0] == j[0] and i[1] == j[1] and i[2] == j[2]) or (
        #                         i[0] == j[1] and i[1] == j[0] and i[2] == j[2]):
        #                     self.overlap_count[k]['correct'] += 1

        for i in gold_pairs:
            for j in pred_pairs:
                if (i[0] == j[0] and i[1] == j[1] and i[2] == j[2]) or (
                        i[0] == j[1] and i[1] == j[0] and i[2] == j[2]):
                    self.correct_triplet += 1

                if (i[0] == j[0] and i[1] == j[1]) or (
                        i[0] == j[1] and i[1] == j[0]):
                    self.correct_pair += 1

    def eval_action(self, pred_actions, gold_actions):
        """

        :param pred_actions:
        :param gold_actions:
        :return:
        """
        pred = np.array([self.action_vocab[x] for x in pred_actions])
        gold = np.array([self.action_vocab[x] for x in gold_actions])
        for x in range(len(self.action_vocab)):
            self.act_count[x]['pred'] += np.sum(pred == x)
            self.act_count[x]['gold'] += np.sum(gold == x)
        for p, g in zip(pred, gold):
            if p == g:
                self.act_count[p]['correct'] += 1
            self.act_count[g]['transition'][p] += 1
        self.pred_ections.extend(pred)
        self.gold_actions.extend(gold)

    def report_triplet(self):
        p_aspect = self.correct_aspects / (self.num_pre_aspects + 1e-18)
        r_aspect = self.correct_aspects / (self.num_gold_aspects + 1e-18)
        f1_aspect = 2 * p_aspect * r_aspect / (p_aspect + r_aspect + 1e-18)

        p_opinion = self.correct_opinions / (self.num_pre_opinions + 1e-18)
        r_opinion = self.correct_opinions / (self.num_gold_opinions + 1e-18)
        f1_opinion = 2 * p_opinion * r_opinion / (p_opinion + r_opinion + 1e-18)

        p_triplet = self.correct_triplet / (self.num_pre_pair + 1e-18)
        r_triplet = self.correct_triplet / (self.num_gold_pair + 1e-18)
        f1_triplet = 2 * p_triplet * r_triplet / (p_triplet + r_triplet + 1e-18)

        # p_pair_role = self.correct_pair_with_role / (self.num_pre_pair + 1e-18)
        # r_pair_role = self.correct_pair_with_role / (self.num_gold_pair + 1e-18)
        # f_pair_role = 2 * p_pair_role * r_pair_role / (p_pair_role + r_pair_role + 1e-18)
        for x in range(len(self.action_vocab)):
            self.act_count[x]['p'] = round(self.act_count[x]['correct'] / (self.act_count[x]['pred'] + 1e-18), 4)
            self.act_count[x]['r'] = round(self.act_count[x]['correct'] / (self.act_count[x]['gold'] + 1e-18), 4)
            self.act_count[x]['f1'] = round(2 * self.act_count[x]['p'] * self.act_count[x]['r'] / (self.act_count[x]['p'] + self.act_count[x]['r'] + 1e-18), 4)

        for k, v in self.overlap_count.items():
            self.overlap_count[k]['p'] = round(self.overlap_count[k]['correct'] / (self.overlap_count[k]['pred'] + 1e-18), 4)
            self.overlap_count[k]['r'] = round(self.overlap_count[k]['correct'] / (self.overlap_count[k]['gold'] + 1e-18), 4)
            self.overlap_count[k]['f1'] = round(2 * self.overlap_count[k]['p'] * self.overlap_count[k]['r'] / (self.overlap_count[k]['p'] + self.overlap_count[k]['r'] + 1e-18), 4)

        return (p_aspect, r_aspect, f1_aspect), (p_opinion, r_opinion, f1_opinion), (p_triplet, r_triplet, f1_triplet), self.act_count, self.overlap_count

    def report_pair(self):
        p_aspect = self.correct_aspects / (self.num_pre_aspects + 1e-18)
        r_aspect = self.correct_aspects / (self.num_gold_aspects + 1e-18)
        f1_aspect = 2 * p_aspect * r_aspect / (p_aspect + r_aspect + 1e-18)

        p_opinion = self.correct_opinions / (self.num_pre_opinions + 1e-18)
        r_opinion = self.correct_opinions / (self.num_gold_opinions + 1e-18)
        f1_opinion = 2 * p_opinion * r_opinion / (p_opinion + r_opinion + 1e-18)

        p_pair = self.correct_pair / (self.num_pre_pair + 1e-18)
        r_pair = self.correct_pair / (self.num_gold_pair + 1e-18)
        f1_pair = 2 * p_pair * r_pair / (p_pair + r_pair + 1e-18)

        # p_pair_role = self.correct_pair_with_role / (self.num_pre_pair + 1e-18)
        # r_pair_role = self.correct_pair_with_role / (self.num_gold_pair + 1e-18)
        # f_pair_role = 2 * p_pair_role * r_pair_role / (p_pair_role + r_pair_role + 1e-18)
        for x in range(len(self.action_vocab)):
            self.act_count[x]['p'] = round(self.act_count[x]['correct'] / (self.act_count[x]['pred'] + 1e-18), 4)
            self.act_count[x]['r'] = round(self.act_count[x]['correct'] / (self.act_count[x]['gold'] + 1e-18), 4)
            self.act_count[x]['f1'] = round(2 * self.act_count[x]['p'] * self.act_count[x]['r'] / (self.act_count[x]['p'] + self.act_count[x]['r'] + 1e-18), 4)

        for k, v in self.overlap_count.items():
            self.overlap_count[k]['p'] = round(self.overlap_count[k]['correct'] / (self.overlap_count[k]['pred'] + 1e-18), 4)
            self.overlap_count[k]['r'] = round(self.overlap_count[k]['correct'] / (self.overlap_count[k]['gold'] + 1e-18), 4)
            self.overlap_count[k]['f1'] = round(2 * self.overlap_count[k]['p'] * self.overlap_count[k]['r'] / (self.overlap_count[k]['p'] + self.overlap_count[k]['r'] + 1e-18), 4)

        return (p_aspect, r_aspect, f1_aspect), (p_opinion, r_opinion, f1_opinion), (p_pair, r_pair, f1_pair), self.act_count, self.overlap_count

    def get_coref_ent(self, g_ent_typed):
        ent_ref_dict = {}
        for ent1 in g_ent_typed:
            start1, end1, ent_type1, ent_ref1 = ent1
            coref_ents = []
            ent_ref_dict[(start1, end1)] = coref_ents
            for ent2 in g_ent_typed:
                start2, end2, ent_type2, ent_ref2 = ent2
                if ent_ref1 == ent_ref2:
                    coref_ents.append((start2, end2))
        return ent_ref_dict

    def split_prob(self, pred_args):
        sp_args, probs = [], []
        for arg in pred_args:
            sp_args.append(arg[:-1])
            probs.append(arg[-1])
        return sp_args, probs

    def get_key(self, dict, value):
        return [k for k, v in dict.items() if v == value]

    def get_pair(self, pair, a_dict, o_dict, k):

        try:
            v = a_dict[k] if k in a_dict.keys() else o_dict[k]
        except:
            print(pair)
            print(a_dict)
            print(o_dict)
            print(k)
        res = []
        for p in pair:
            if v in p[:2]:
                res.append(p)
        return res

    def get_overlap(self, pairs, a_dict, o_dict):
        pair_term = []
        overlap_num = 0
        for p in pairs:
            t1, t2 = p[0], p[1]
            pair_term.extend(self.get_key(a_dict, t1))
            pair_term.extend(self.get_key(o_dict, t2))
        res = Counter(pair_term)
        overlap_dict = {}
        for p in pairs:
            a_key = self.get_key(a_dict, p[0])
            o_key = self.get_key(o_dict, p[1])
            c0 = res[a_key[0]]
            c1 = res[o_key[0]]
            if c1 > 1 or c0 > 1:
                overlap_num += 1
            if c0 in overlap_dict.keys():
                if p not in overlap_dict[c0]:
                    overlap_dict[c0].append(p)
            else:
                overlap_dict[c0] = [p]
            if c1 in overlap_dict.keys():
                if p not in overlap_dict[c1]:
                    overlap_dict[c1].append(p)
            else:
                overlap_dict[c1] = [p]

        return overlap_dict, overlap_num
