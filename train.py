import os
import math

from model import *

from io_utils import read_pickle, write_lines, read_lines, write_texts
from vocab import Vocab
from eval import AOPEval


class Trainer(object):

    def __init__(self):
        logger.info('Loading data...')
        self.train_list, self.dev_list, self.test_list, self.token_vocab, self.char_vocab, \
        self.action_vocab, self.polarity_vocab, self.pos_vocab, self.dep_type_vocab = read_pickle(data_config['inst_pl_file'])

        logger.info(
            'Sent size train: %d, dev: %d, test:%d' % (len(self.train_list), len(self.dev_list), len(self.test_list)))

        logger.info('Loading pretrained from: %s' % data_config['vec_npy'])
        pretrained_vec = np.load(data_config['vec_npy'])


        self.unk_idx = self.token_vocab[Vocab.UNK]
        joint_config['n_chars'] = len(self.char_vocab)
        self.trans_model = MainModel(self.token_vocab.get_vocab_size(),
                                     self.action_vocab,
                                     self.polarity_vocab,
                                     pos_dict=self.pos_vocab,
                                     dep_type_vocab=self.dep_type_vocab,
                                     pretrained_vec=pretrained_vec,
                                     l2_norm=joint_config['l2_norm'])
        logger.info("Model:%s" % type(self.trans_model))

        overlap_type = [1, 2, 3, 4, 5, 6, 13]
        self.aop_eval = AOPEval(self.action_vocab, overlap_type=overlap_type)

    def unk_replace_singleton(self, unk_idx, unk_ratio, words):
        noise = words[:]
        bernoulli = np.random.binomial(n=1, p=unk_ratio, size=len(words))
        for i, idx in enumerate(words):
            if self.token_vocab.is_singleton(idx) and bernoulli[i] == 1:
                noise[i] = unk_idx
        return noise

    def iter_batch(self, inst_list, shuffle=True):
        batch_size = joint_config['batch_size']
        if shuffle:
            random.shuffle(inst_list)
        inst_len = len(inst_list)
        plus_n = 0 if (inst_len % batch_size) == 0 else 1
        num_batch = (len(inst_list) // batch_size) + plus_n

        start = 0
        for i in range(num_batch):
            batch_inst = inst_list[start: start + batch_size]
            start += batch_size
            yield batch_inst

    def train_batch(self):
        loss_all = 0.
        batch_num = 0
        for batch_inst in self.iter_batch(self.train_list, shuffle=True):
            dy.renew_cg()
            loss_minibatch = []

            for inst in batch_inst:
                words = inst['word_indices']
                if joint_config['unk_ratio'] > 0:
                    words = self.unk_replace_singleton(self.unk_idx, joint_config['unk_ratio'], words)
                loss_rep = self.trans_model(words, inst['char_indices'],
                                            inst['action_indices'], inst['actions'],
                                            inst['sent_range'], roles=inst['pair_idx'],
                                            pos_list=inst['pos_indices'],
                                            dep_list=inst['dep_label_indices'],
                                            dep_head=inst['dep_head'])

                loss_minibatch.append(loss_rep)

            batch_loss = dy.esum(loss_minibatch) / len(loss_minibatch)
            # batch_loss += dy.l2_norm(dy.parameter(self.trans_model.model))
            loss_all += batch_loss.value()
            batch_loss.backward()
            self.trans_model.update()
            batch_num += 1

        logger.info('loss %.5f ' % (loss_all / float(batch_num)))

    def eval(self, inst_list, write_ent_file=None, is_write_ent=False, mtype='dev'):
        self.aop_eval.reset()
        sent_num_eval = 0

        ent_lines = []
        for inst in inst_list:
            _, pred_pairs, aspect_terms, opinion_terms, pred_action_strs = self.trans_model.decode(
                inst['word_indices'], inst['char_indices'], inst['action_indices'],
                inst['actions'], inst['sent_range'], pos_list=inst['pos_indices'],
                dep_list=inst['dep_label_indices'],
                dep_head=inst['dep_head'],  mtype=mtype)

            self.aop_eval.update(aspect_terms, inst['aspects_idx'],  # list
                                 opinion_terms, inst['opinions_idx'],  # list
                                 pred_pairs, inst['pair_idx'],
                                 inst['overlap'],  # triplet
                                 eval_arg=True)

            ent_line = str(sent_num_eval) + ' \n'

            ent_line += 'gold:\t '
            ent_str_list = []
            for ent in inst['pair_idx']:
                ent_str_list.append("<"+str(ent[0]) + ', ' + str(ent[1]) + ', ' + self.polarity_vocab.get_token(ent[2]) + ">")
            ent_line += ' , '.join(ent_str_list)

            ent_line += '\npredicted:\t '
            ent_str_list = []
            for ent in pred_pairs:
                ent_str_list.append("<"+str(ent[0]) + ', ' + str(ent[1]) + ', ' + self.polarity_vocab.get_token(ent[2]) + ">")
            ent_line += ' , '.join(ent_str_list)

            ent_line += ' \n\n'
            ent_lines.append(ent_line)
            sent_num_eval += 1

        if write_ent_file is not None and is_write_ent:
            write_texts(write_ent_file, ent_lines)

    def train(self, save_model=True):
        logger.info(joint_config['msg_info'])
        best_f1_aspect, best_f1_opinion, best_f1_pair = 0., 0., 0.
        best_epoch = 0

        adjust_lr = False
        stop_patience = joint_config['patience']
        stop_count = 0
        eval_best_arg = True  # for other task than event
        t_cur = 1
        t_i = 4
        t_mul = 2
        lr_max, lr_min = joint_config['init_lr'], joint_config['minimum_lr']
        for epoch in range(joint_config['num_epochs']):
            anneal_lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t_cur / t_i))
            self.trans_model.set_lr(anneal_lr)

            logger.info('--------------------------------------')
            logger.info('Epoch : %d' % (epoch+1))
            logger.info('LR : %.5f' % self.trans_model.get_lr())

            self.train_batch()

            self.eval(self.dev_list, write_ent_file=data_config['write_eval_file_dev'], is_write_ent=True, mtype='dev')
            (p_aspect, r_aspect, f1_aspect), (p_opinion, r_opinion, f1_opinion), (p_pair, r_pair, f1_pair), report, overlap = self.aop_eval.report_pair()

            logger.info('Aspect Extraction  P:%.5f, R:%.5f, F:%.5f' % (p_aspect, r_aspect, f1_aspect))
            logger.info('Opinion Extraction P:%.5f, R:%.5f, F:%.5f' % (p_opinion, r_opinion, f1_opinion))
            logger.info('Pair Extraction    P:%.5f, R:%.5f, F:%.5f' % (p_pair, r_pair, f1_pair))

            if t_cur == t_i:
                t_cur = 0
                t_i *= t_mul

            t_cur += 1

            if not eval_best_arg:
                continue
            if f1_pair >= best_f1_pair:
                best_f1_aspect = f1_aspect
                best_f1_opinion = f1_opinion
                best_f1_pair = f1_pair
                best_epoch = epoch

                stop_count = 0

                if save_model:
                    logger.info('Saving model %s' % data_config['model_save_file'])
                    self.trans_model.save_model(data_config['model_save_file'])

            else:
                stop_count += 1
                if stop_count >= stop_patience:
                    logger.info('Stop training, Arg performance did not improved for %d epochs' % stop_count)
                    break

                if adjust_lr:
                    self.trans_model.decay_lr(joint_config['decay_lr'])
                    logger.info('@@@@  Adjusting LR: %.5f  @@@@@@' % self.trans_model.get_lr())

                if self.trans_model.get_lr() < joint_config['minimum_lr']:
                    adjust_lr = False

            best_msg = '*****Best epoch: %d aspect, opinion and pair F:%.5f, F:%.5f F:%.5f******' % (best_epoch+1,
                                                                                                     best_f1_aspect,
                                                                                                     best_f1_opinion,
                                                                                                     best_f1_pair)
            logger.info(best_msg)

        return best_msg, best_f1_pair

    def test(self, fname):
        self.trans_model.load_model(fname)
        self.eval(self.test_list, write_ent_file=data_config['write_eval_file_test'], is_write_ent=True, mtype='test')
        (p_aspect, r_aspect, f1_aspect), (p_opinion, r_opinion, f1_opinion), (p_pair, r_pair, f1_pair), report, overlap = self.aop_eval.report_pair()

        logger.info('Aspect Extraction  P:%.5f, R:%.5f, F:%.5f' % (p_aspect, r_aspect, f1_aspect))
        logger.info('Opinion Extraction P:%.5f, R:%.5f, F:%.5f' % (p_opinion, r_opinion, f1_opinion))
        logger.info('Pair Extraction    P:%.5f, R:%.5f, F:%.5f' % (p_pair, r_pair, f1_pair))


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train(save_model=True)

    logger.info('---------------Test Results---------------')
    ckp_path = data_config['model_save_file']
    trainer.test(ckp_path)
