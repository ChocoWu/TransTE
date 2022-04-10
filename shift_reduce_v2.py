import numpy as np
import dynet as dy
import nn
import ops
from dy_utils import ParamManager as pm
from actions import Actions
from vocab import Vocab
# from event_constraints import EventConstraint
import io_utils


class RoleLabeler(object):

    def __init__(self, config, encoder_output_dim, action_dict, role_type_dict):
        self.config = config
        self.model = pm.global_collection()
        bi_rnn_dim = encoder_output_dim  # config['rnn_dim'] * 2 #+ config['edge_embed_dim']
        lmda_dim = config['lmda_rnn_dim']

        self.lmda_dim = lmda_dim
        self.bi_rnn_dim = bi_rnn_dim
        self.role_type_dict = role_type_dict

        hidden_input_dim = lmda_dim * 2 + bi_rnn_dim * 2 + config['out_rnn_dim']
        # self.role_attn_hidden = nn.Linear(hidden_input_dim, config['role_embed_dim'])
        #
        # hidden_input_dim = hidden_input_dim + config['role_embed_dim']
        self.hidden_arg = nn.Linear(hidden_input_dim+len(role_type_dict), config['output_hidden_dim'],
                                    activation='tanh')

        self.output_arg = nn.Linear(config['output_hidden_dim'], len(role_type_dict))

        hidden_input_dim_co = lmda_dim * 3 + bi_rnn_dim * 2 + config['out_rnn_dim']
        self.hidden_ent_corel = nn.Linear(hidden_input_dim_co, config['output_hidden_dim'],
                                          activation='tanh')
        self.output_ent_corel = nn.Linear(config['output_hidden_dim'], 2)

        self.position_embed = nn.Embedding(500, 20)

        attn_input = self.bi_rnn_dim * 1 + 20 * 2
        self.attn_hidden = nn.Linear(attn_input, 80, activation='tanh')
        self.attn_out = nn.Linear(80, 1)

        self.distrib_attn_hidden = nn.Linear(hidden_input_dim + len(role_type_dict), 80, activation='tanh')
        self.distrib_attn_out = nn.Linear(80, 1)
        self.empty_embedding = self.model.add_parameters((len(role_type_dict),), name='stackGuardEmb')

    def arg_prd_distributions_role_attn(self, inputs, arg_prd_distributions_role):
        inputs_ = [inputs for _ in range(len(arg_prd_distributions_role))]
        arg_prd_distributions_role = ops.cat(arg_prd_distributions_role, 1)
        inputs_ = ops.cat(inputs_, 1)
        att_input = dy.concatenate([arg_prd_distributions_role, inputs_], 0)
        hidden = self.distrib_attn_hidden(att_input)
        attn_out = self.distrib_attn_out(hidden)
        attn_prob = nn.softmax(attn_out, dim=1)
        rep = arg_prd_distributions_role * dy.transpose(attn_prob)
        return rep

    def opinion_aspect_role_attn(self, inputs, role_table=None):
        role_list = []
        for k, v in self.role_type_dict.__iter__():
            role_list.append(k)
        role_emb = ops.cat(role_table(role_list), 1)
        hidden = self.role_attn_hidden(inputs)
        # attn = nn.dot_transpose(hidden, role_emb)
        attn = dy.transpose(hidden) * role_emb
        attn_output = role_emb * dy.transpose(attn)
        # attn_output = nn.dot_transpose(attn, role_emb)
        return attn_output

    def forward(self, beta_embed, lmda_embed, sigma_embed, alpha_embed, out_embed, gold_role_label, role_table=None,
                history_info=None):

        # attn_rep = self.position_aware_attn(hidden_mat, last_h, prd_idx, prd_idx, arg_idx, arg_idx, seq_len)

        state_embed = ops.cat([beta_embed, lmda_embed, sigma_embed, alpha_embed, out_embed], dim=0)

        if history_info is not None:
            if len(history_info) > 1:
                rep = ops.cat([self.arg_prd_distributions_role_attn(state_embed, history_info), state_embed], 0)
            else:
                rep = ops.cat([self.empty_embedding, state_embed], 0)
        else:
            rep = ops.cat([self.empty_embedding, state_embed], 0)

        rep = dy.dropout(rep, 0.25)
        hidden = self.hidden_arg(rep)
        out = self.output_arg(hidden)

        loss = dy.pickneglogsoftmax(out, gold_role_label)
        return loss

    def decode(self, beta_embed, lmda_embed, sigma_embed, alpha_embed, out_embed, role_table=None, history_info=None):
        # attn_rep = self.position_aware_attn(hidden_mat, last_h, prd_idx, prd_idx, arg_idx, arg_idx, seq_len)

        state_embed = ops.cat([beta_embed, lmda_embed, sigma_embed, alpha_embed, out_embed], dim=0)
        # if role_table is not None:
        #     rep = ops.cat([self.opinion_aspect_role_attn(state_embed, role_table), state_embed, self.empty_embedding], 0)
        # rep = ops.cat([self.empty_embedding, state_embed], 0)

        # if len(history_info) > 1:
        if history_info is not None:
            if len(history_info) > 1:
                rep = ops.cat([self.arg_prd_distributions_role_attn(state_embed, history_info), state_embed], 0)
            else:
                rep = ops.cat([self.empty_embedding, state_embed], 0)
        else:
            rep = ops.cat([self.empty_embedding, state_embed], 0)

        hidden = self.hidden_arg(rep)
        out = self.output_arg(hidden)
        np_score = out.npvalue().flatten()
        return np.argmax(np_score)

    def position_aware_attn(self, hidden_mat, last_h, start1, ent1, start2, end2, seq_len):
        tri_pos_list = []
        ent_pos_list = []

        for i in range(seq_len):
            tri_pos_list.append(io_utils.relative_position(start1, ent1, i))
            ent_pos_list.append(io_utils.relative_position(start2, end2, i))

        tri_pos_emb = self.position_embed(tri_pos_list)
        tri_pos_mat = ops.cat(tri_pos_emb, 1)
        ent_pos_emb = self.position_embed(ent_pos_list)
        ent_pos_mat = ops.cat(ent_pos_emb, 1)

        att_input = ops.cat([hidden_mat, tri_pos_mat, ent_pos_mat], 0)

        hidden = self.attn_hidden(att_input)
        attn_out = self.attn_out(hidden)

        attn_prob = nn.softmax(attn_out, dim=1)

        rep = hidden_mat * dy.transpose(attn_prob)

        return rep


class ShiftReduce(object):

    def __init__(self, config, encoder_output_dim, action_dict, role_type_dict):

        self.config = config
        self.model = pm.global_collection()

        # self.role_labeler = RoleLabeler(config, encoder_output_dim, action_dict, role_type_dict)
        # self.role_null_id = role_type_dict[Vocab.NULL]
        # self.role_type_dict = role_type_dict

        bi_rnn_dim = encoder_output_dim  # config['rnn_dim'] * 2 #+ config['edge_embed_dim']
        lmda_dim = config['lmda_rnn_dim']

        self.lmda_dim = lmda_dim
        self.bi_rnn_dim = bi_rnn_dim

        dp_state = config['dp_state']
        dp_state_h = config['dp_state_h']

        # ------ states
        self.gamma_var = nn.LambdaVar(lmda_dim)
        self.sigma_a_rnn = nn.StackLSTM(lmda_dim, lmda_dim, dp_state, dp_state_h)
        self.alpha_a_rnn = nn.StackLSTM(lmda_dim, lmda_dim, dp_state, dp_state_h)
        self.sigma_p_rnn = nn.StackLSTM(lmda_dim, lmda_dim, dp_state, dp_state_h)
        self.alpha_p_rnn = nn.StackLSTM(lmda_dim, lmda_dim, dp_state, dp_state_h)
        self.term_rnn = nn.StackLSTM(lmda_dim, lmda_dim, dp_state, dp_state_h)

        self.actions_rnn = nn.StackLSTM(config['action_embed_dim'], config['action_rnn_dim'], dp_state, dp_state_h)
        self.out_rnn = nn.StackLSTM(bi_rnn_dim, config['out_rnn_dim'], dp_state, dp_state_h)

        # ------ states

        self.act_table = nn.Embedding(len(action_dict), config['action_embed_dim'])
        self.role_table = nn.Embedding(len(role_type_dict), config['role_embed_dim'])

        self.act = Actions(action_dict, role_type_dict)

        hidden_input_dim = bi_rnn_dim + lmda_dim * 6 \
                           + config['action_rnn_dim'] + config['out_rnn_dim']

        self.hidden_linear = nn.Linear(hidden_input_dim, config['output_hidden_dim'], activation='tanh')
        self.output_linear = nn.ActionGenerator(config['output_hidden_dim'],  len(action_dict))

        prd_embed_dim = config['prd_embed_dim']

        prd_to_lmda_dim = bi_rnn_dim + prd_embed_dim  # + config['sent_vec_dim']
        self.prd_to_lmda = nn.Linear(prd_to_lmda_dim, lmda_dim, activation='tanh')

        self.arg_op_distrib_as = nn.Linear(lmda_dim * 2, lmda_dim, activation='softmax')
        self.arg_as_distrib_op = nn.Linear(lmda_dim * 2, lmda_dim, activation='softmax')

        # beta
        self.empty_buffer_emb = self.model.add_parameters((bi_rnn_dim,), name='bufferGuardEmb')

    def __call__(self, toks, hidden_state_list, last_h, oracle_actions=None,
                 oracle_action_strs=None, is_train=True, prds=None, roles=None):

        # def get_role_label(sigma_last_idx, lmda_idx):
        #     for i in roles:
        #         if (i[0] == sigma_last_idx and i[1] == lmda_idx) or (i[1] == sigma_last_idx and i[0] == lmda_idx):
        #             return i[2]
        #     return self.role_null_id
            # raise RuntimeError('Unknown aspect term & opinion term idx: ' + str(sigma_last_idx) + ' ' + str(lmda_idx))

        def get_history(aspect_idx, opinion_idx, history):
            res = []
            e = dy.random_normal(len(self.role_type_dict), mean=0, stddev=1.0, batch_size=1)
            if len(history) == 0:
                res.append(e)
            else:
                for a, p, feat in history:
                    if aspect_idx == a or opinion_idx == p:
                        res.append(feat)
                if len(res) == 0:
                    res.append(e)
            return res

        frames = []
        aspect_terms = []
        opinion_terms = []

        hidden_mat = ops.cat(hidden_state_list, 1)
        seq_len = len(toks)

        # beta, queue, for candidate sentence.
        buffer = nn.Buffer(self.bi_rnn_dim, hidden_state_list)

        losses = []
        loss_roles = []
        pred_action_strs = []

        # storage the triplet feature
        history_tri_feat = []

        self.sigma_a_rnn.init_sequence(not is_train)
        self.alpha_a_rnn.init_sequence(not is_train)
        self.sigma_p_rnn.init_sequence(not is_train)
        self.alpha_p_rnn.init_sequence(not is_train)
        self.term_rnn.init_sequence(not is_train)
        self.actions_rnn.init_sequence(not is_train)
        self.out_rnn.init_sequence(not is_train)

        steps = 0
        # while not (buffer.is_empty() and self.gamma_var.is_empty() and self.term_rnn.is_empty()) and steps < len(oracle_actions):
        while True:
            if steps >= len(oracle_actions):
                break
            if buffer.idx >= seq_len:
                break
            if buffer.is_empty() and self.gamma_var.is_empty():
                break
            # 上一个action
            pre_action = None if self.actions_rnn.is_empty() else self.actions_rnn.last_idx()

            # based on parser state, get valid actions.
            # only a very small subset of actions are valid, as below.
            valid_actions = []
            # if sigma_rnn_empty_flag == 1:
            #     valid_actions += [self.act.shift_id]
            if buffer.is_empty() and self.gamma_var.is_empty():
                valid_actions += [self.act.term_gen_a_id, self.act.term_gen_o_id]
            elif pre_action is not None and self.act.is_delete(pre_action):
                valid_actions += [self.act.delete_id, self.act.term_shift_a_id, self.act.term_shift_o_id]
            elif pre_action is not None and self.act.is_shift(pre_action):
                valid_actions += [self.act.delete_id, self.act.term_shift_o_id, self.act.term_shift_a_id]
            elif pre_action is not None and self.act.is_arc(pre_action):
                valid_actions += [self.act.arc_id, self.act.no_arc_id, self.act.shift_id]
            elif pre_action is not None and self.act.is_no_arc(pre_action):
                valid_actions += [self.act.arc_id, self.act.no_arc_id, self.act.shift_id]
            elif pre_action is not None and self.act.is_term_shift_o(pre_action):
                valid_actions += [self.act.term_shift_o_id, self.act.term_gen_o_id]
            elif pre_action is not None and self.act.is_term_gen_o(pre_action):
                valid_actions += [self.act.term_back_id, self.act.arc_id]
            elif pre_action is not None and self.act.is_term_shift_a(pre_action):
                valid_actions += [self.act.term_shift_a_id, self.act.term_gen_a_id]
            elif pre_action is not None and self.act.is_term_gen_a(pre_action):
                valid_actions += [self.act.term_back_id, self.act.arc_id, self.act.no_arc_id]
            elif pre_action is not None and self.act.is_term_back(pre_action):  # term_back
                valid_actions += [self.act.shift_id, self.act.arc_id, self.act.no_arc_id]
            elif self.sigma_a_rnn.is_empty() and (not self.alpha_a_rnn.is_empty() or not self.gamma_var.is_empty()):
                valid_actions += [self.act.shift_id]
            elif self.sigma_p_rnn.is_empty() and (not self.alpha_p_rnn.is_empty() or not self.gamma_var.is_empty()):
                valid_actions += [self.act.shift_id]
            else:
                valid_actions += [self.act.delete_id, self.act.term_shift_o_id, self.act.term_shift_a_id,
                                  self.act.no_arc_id, self.act.term_back_id, self.act.arc_id]

            # predicting action
            beta_embed = self.empty_buffer_emb if buffer.is_empty() else buffer.hidden_embedding()
            lmda_embed = self.gamma_var.embedding()
            sigma_a_embed = self.sigma_a_rnn.embedding()
            alpha_a_embed = self.alpha_a_rnn.embedding()
            sigma_p_embed = self.sigma_p_rnn.embedding()
            alpha_p_embed = self.alpha_p_rnn.embedding()
            term_embed = self.term_rnn.embedding()
            action_embed = self.actions_rnn.embedding()
            out_embed = self.out_rnn.embedding()

            state_embed = ops.cat([beta_embed, lmda_embed, sigma_a_embed, sigma_p_embed, alpha_a_embed, alpha_p_embed,
                                   term_embed, action_embed, out_embed], dim=0)
            if is_train:
                state_embed = dy.dropout(state_embed, self.config['dp_out'])

            hidden_rep = self.hidden_linear(state_embed)

            logits = self.output_linear(hidden_rep)
            if is_train:
                log_probs = dy.log_softmax(logits, valid_actions)
            else:
                log_probs = dy.log_softmax(logits, valid_actions)

            if is_train:
                action = oracle_actions[steps]
                action_str = oracle_action_strs[steps]
                if action not in valid_actions:
                    raise RuntimeError('Action %s dose not in valid_actions, %s(pre) %s: [%s]' % (
                    action_str, self.act.to_act_str(pre_action),
                    self.act.to_act_str(action), ','.join(
                        [self.act.to_act_str(ac) for ac in valid_actions])))
                losses.append(dy.pick(log_probs, action))
            else:
                np_log_probs = log_probs.npvalue()
                act_prob = np.max(np_log_probs)
                action = np.argmax(np_log_probs)
                action_str = self.act.to_act_str(action)
                pred_action_strs.append(action_str)

            # if True:continue
            # update the parser state according to the action.
            if self.act.is_delete(action):
                # pop the word wi off buffer
                hx, idx = buffer.pop()
                self.out_rnn.push(hx, idx)

            elif self.act.is_shift(action):
                # while no elements are in sigma
                while not self.alpha_a_rnn.is_empty():
                    self.sigma_a_rnn.push(*self.alpha_a_rnn.pop())
                while not self.alpha_p_rnn.is_empty():
                    self.sigma_p_rnn.push(*self.alpha_p_rnn.pop())
                while not self.gamma_var.is_empty():
                    if self.gamma_var.lambda_type == 'opinion':
                        hx, idx = self.gamma_var.pop()
                        self.sigma_p_rnn.push(hx, idx)
                    elif self.gamma_var.lambda_type == 'aspect':
                        hx, idx = self.gamma_var.pop()
                        self.sigma_a_rnn.push(hx, idx)
                    else:
                        raise RuntimeError('Wrong lambda type, not aspect or opinion')
            elif self.act.is_term_shift_o(action) or self.act.is_term_shift_a(action):
                # move the top word wi from buffer to term
                if not buffer.is_empty():
                    hx, idx = buffer.pop()
                    self.term_rnn.push(hx, idx)
            elif self.act.is_term_gen_o(action):
                # summarize all elements in term_rnn to an opinion vector representation and copy it to gamma_var
                vec = []
                opinion_idx = []
                start_idx = 0
                for i in self.term_rnn.iter():
                    start_idx = i[1]
                    vec.append(i[0])
                    opinion_idx.append(i[1])
                # need summarize operation for vec
                vec = ops.sum(vec)
                self.gamma_var.push(vec, opinion_idx, nn.LambdaVar.OPINION)
                opinion_terms.append(opinion_idx)
            elif self.act.is_term_gen_a(action):
                # summarize all elements in term_rnn to an aspect vector representation and copy it to gamma_var
                vec = []
                aspect_idx = []
                for i in self.term_rnn.iter():
                    vec.append(i[0])
                    aspect_idx.append(i[1])
                # need summarize operation for vec
                vec = ops.sum(vec)
                self.gamma_var.push(vec, aspect_idx, nn.LambdaVar.ASPECT)
                aspect_terms.append(aspect_idx)
            elif self.act.is_arc(action):
                lmda_idx = self.gamma_var.idx
                lmda_embed = self.gamma_var.embedding()
                lmda_type = self.gamma_var.lambda_type
                if lmda_type == 'opinion':
                    if not self.sigma_a_rnn.is_empty():
                        sigma_last_embed, sigma_last_idx = self.sigma_a_rnn.pop()
                        # according to the current aspect and opinion to
                        # obtain historical information that has been predicted
                        # history = get_history(sigma_last_idx, lmda_idx, history_tri_feat)
                        # if is_train:
                        #     # role_label = get_role_label(sigma_last_idx, lmda_idx)
                        #     # loss_role = self.role_labeler.forward(beta_embed, lmda_embed, sigma_last_embed,
                        #     #                                       alpha_a_embed, out_embed, role_label,
                        #     #                                       role_table=self.role_table, history_info=history)
                        #     # loss_roles.append(loss_role)
                        # else:
                        #     role_label = self.role_labeler.decode(beta_embed, lmda_embed, sigma_last_embed,
                        #                                           alpha_a_embed, out_embed,
                        #                                           role_table=self.role_table, history_info=history)
                        #
                        # polarity_emb = self.role_table[role_label]
                        tri_feature = self.arg_as_distrib_op(ops.cat([sigma_last_embed, lmda_embed], dim=0))
                        history_tri_feat.append((sigma_last_idx, lmda_idx, tri_feature))

                        self.alpha_a_rnn.push(sigma_last_embed, sigma_last_idx)

                        frame = (sigma_last_idx, lmda_idx, 1)
                        frames.append(frame)
                        sigma_rnn_empty_flag = 0
                    else:
                        sigma_rnn_empty_flag = 1
                elif lmda_type == 'aspect':
                    if not self.sigma_p_rnn.is_empty():
                        sigma_last_embed, sigma_last_idx = self.sigma_p_rnn.pop()
                        # history = get_history(lmda_idx, sigma_last_idx, history_tri_feat)
                        # if is_train:
                        #     role_label = get_role_label(sigma_last_idx, lmda_idx)
                        #     loss_role = self.role_labeler.forward(beta_embed, lmda_embed, sigma_last_embed,
                        #                                           alpha_p_embed, out_embed, role_label,
                        #                                           role_table=self.role_table, history_info=history)
                        #     loss_roles.append(loss_role)
                        # else:
                        #     role_label = self.role_labeler.decode(beta_embed, lmda_embed, sigma_last_embed,
                        #                                           alpha_p_embed, out_embed,
                        #                                           role_table=self.role_table, history_info=history)
                        # polarity_emb = self.role_table[role_label]
                        tri_feature = self.arg_op_distrib_as(ops.cat([sigma_last_embed, lmda_embed], dim=0))
                        history_tri_feat.append((lmda_idx, sigma_last_idx, tri_feature))

                        self.alpha_p_rnn.push(sigma_last_embed, sigma_last_idx)

                        frame = (lmda_idx, sigma_last_idx, 1)
                        frames.append(frame)
                        sigma_rnn_empty_flag = 0
                    else:
                        sigma_rnn_empty_flag = 1
                else:
                    raise RuntimeError('Wrong lambda type, not aspect or opinion')

            elif self.act.is_no_arc(action):
                # alpha holding elements temporarily popped out of sigma
                lmda_type = self.gamma_var.lambda_type
                if lmda_type == 'opinion':
                    if not self.sigma_a_rnn.is_empty():
                        self.alpha_a_rnn.push(*self.sigma_a_rnn.pop())
                        sigma_rnn_empty_flag = 0
                    else:
                        sigma_rnn_empty_flag = 1
                elif lmda_type == 'aspect':
                    if not self.sigma_p_rnn.is_empty():
                        self.alpha_p_rnn.push(*self.sigma_p_rnn.pop())
                        sigma_rnn_empty_flag = 0
                    else:
                        sigma_rnn_empty_flag = 1
                else:
                    raise RuntimeError('Wrong lambda type, not aspect or opinion')

            elif self.act.is_term_back(action):
                start_idx = 0
                while not self.term_rnn.is_empty():
                    _, start_idx = self.term_rnn.pop()
                # the elements in buffer don't pop operation, so there is no need to push operation,
                # only need to change the idx
                buffer.move_pointer(start_idx + 1)
            else:
                raise RuntimeError('Unknown action type:' + str(action) + self.act.to_act_str(action))

            self.actions_rnn.push(self.act_table[action], action)

            steps += 1

        self.clear()

        return losses, loss_roles, frames, aspect_terms, opinion_terms, pred_action_strs

    def clear(self):
        self.sigma_a_rnn.clear()
        self.sigma_p_rnn.clear()
        self.alpha_a_rnn.clear()
        self.alpha_p_rnn.clear()
        self.term_rnn.clear()
        self.actions_rnn.clear()
        self.gamma_var.clear()
        self.out_rnn.clear()

    def same(self, args):
        same_event_ents = set()
        for arg1 in args:
            ent_start1, ent_end1, tri_idx1, _ = arg1
            for arg2 in args:
                ent_start2, ent_end2, tri_idx2, _ = arg2
                if tri_idx1 == tri_idx2:
                    same_event_ents.add((ent_start1, ent_start2))
                    same_event_ents.add((ent_start2, ent_start1))

        return same_event_ents

    def get_valid_args(self, ent_type_id, tri_type_id):
        return self.cached_valid_args[(ent_type_id, tri_type_id)]
