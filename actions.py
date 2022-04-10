# -*- coding: utf-8 -*-


class Actions(object):

    delete = 'DELETE'
    term_shift_o = 'TERM_SHIFT_O'
    term_gen_o = 'TERM_GEN_O'
    term_shift_a = 'TERM_SHIFT_A'
    term_gen_a = 'TERM_GEN_A'
    term_back = 'TERM_BACK'

    shift = 'SHIFT'
    arc = 'ARC'
    no_arc = 'NO_ARC'

    def __init__(self, action_dict, role_type_dict=None):
        self.delete_id = action_dict[Actions.delete]
        self.term_shift_o_id = action_dict[Actions.term_shift_o]
        self.term_gen_o_id = action_dict[Actions.term_gen_o]
        self.term_shift_a_id = action_dict[Actions.term_shift_a]
        self.term_gen_a_id = action_dict[Actions.term_gen_a]
        self.term_back_id = action_dict[Actions.term_back]

        self.shift_id = action_dict[Actions.shift]
        self.arc_id = action_dict[Actions.arc]
        self.no_arc_id = action_dict[Actions.no_arc]

        self.act_id_to_str = {v: k for k, v in action_dict.items()}
        self.act_str_to_id = action_dict

        # for name, id in action_dict.items():
        #     if name.startswith(Actions.left_arc):
        #         self.pair_gen_left_group.add(id)
        #
        #     elif name.startswith(Actions.right_arc):
        #         self.pair_gen_right_group.add(id)

    def to_act_str(self, act_id):
        return self.act_id_to_str[act_id]

    def to_act_id(self, act_str):
        return self.act_str_to_id[act_str]

    # action check
    def is_delete(self, act_id):
        return self.delete_id == act_id

    def is_term_shift_o(self, act_id):
        return self.term_shift_o_id == act_id

    def is_term_gen_o(self, act_id):
        return self.term_gen_o_id == act_id

    def is_term_shift_a(self, act_id):
        return self.term_shift_a_id == act_id

    def is_term_gen_a(self, act_id):
        return self.term_gen_a_id == act_id

    def is_shift(self, act_id):
        return self.shift_id == act_id

    def is_term_back(self, act_id):
        return self.term_back_id == act_id

    def is_arc(self, act_id):
        return self.arc_id == act_id

    def is_no_arc(self, act_id):
        return self.no_arc_id == act_id

    @staticmethod
    def make_oracle(tokens, pairs, aspects, opinions):

        aspects_dic = {str(idx) + '_a': aspect for idx, aspect in enumerate(aspects)}
        opinions_dic = {str(idx) + '_o': opinion for idx, opinion in enumerate(opinions)}

        pair_dic = {}
        a_k, o_k = [], []
        for idx, pair in enumerate(pairs):
            for k, v in aspects_dic.items():
                if pair[0] == v:
                    a_k = k
            for k, v in opinions_dic.items():
                if pair[1] == v:
                    o_k = k
            pair_dic[idx] = (a_k, o_k)

        actions = []

        def is_in_term(dic, idx):
            for k, v in dic.items():
                if idx == int(v[0]):
                    return k, v
            return False

        sent_length = len(tokens)
        idx = 0
        sigma_a, alpha_a, k = [], [], []
        sigma_o, alpha_o, k = [], [], []
        while idx < sent_length:
            if is_in_term(aspects_dic, idx):
                aspect_id, aspect = is_in_term(aspects_dic, idx)
                for a in aspect:
                    aspect_end = a
                    actions.append(Actions.term_shift_a)
                actions.append(Actions.term_gen_a)
                actions.append(Actions.term_back)
                i = 0
                sigma_p_len = len(sigma_o)
                while i < sigma_p_len:
                    pop_id = sigma_o.pop(len(sigma_o) - 1)
                    if (pop_id, aspect_id) in pair_dic.values() or (aspect_id, pop_id) in pair_dic.values():
                        actions.append(Actions.arc)
                        alpha_o.append(pop_id)
                    else:
                        actions.append(Actions.no_arc)
                        alpha_o.append(pop_id)
                    i += 1
                actions.append(Actions.shift)
                sigma_o.extend(alpha_o)
                sigma_a.append(aspect_id)
                alpha_o.clear()
                idx += 1
            elif is_in_term(opinions_dic, idx):
                opinion_id, opinion = is_in_term(opinions_dic, idx)
                for o in opinion:
                    opinion_end = o
                    actions.append(Actions.term_shift_o)
                actions.append(Actions.term_gen_o)
                actions.append(Actions.term_back)
                i = 0
                sigma_a_len = len(sigma_a)
                while i < sigma_a_len:
                    pop_id = sigma_a.pop(len(sigma_a) - 1)
                    if (pop_id, opinion_id) in pair_dic.values() or (opinion_id, pop_id) in pair_dic.values():
                        actions.append(Actions.arc)
                        alpha_a.append(pop_id)
                    else:
                        actions.append(Actions.no_arc)
                        alpha_a.append(pop_id)
                    i += 1
                actions.append(Actions.shift)
                sigma_a.extend(alpha_a)
                sigma_o.append(opinion_id)
                alpha_a.clear()
                idx += 1
            else:
                actions.append(Actions.delete)
                idx += 1

        return actions  







