""" This module will handle the text generation with beam search. """

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import Transformer
from transformer.Beam import Beam

import numpy as np


class Translator(object):
    """ Load with trained model and handle the beam search """

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt.cuda else 'cpu')

        checkpoint = torch.load(opt.model)
        model_opt = checkpoint['settings']
        self.model_opt = model_opt

        model = Transformer(
            model_opt.src_vocab_size,
            model_opt.tgt_vocab_size,
            model_opt.max_token_seq_len,
            self.device,
            tgt_emb_prj_weight_sharing=model_opt.proj_share_weight,
            emb_src_tgt_weight_sharing=model_opt.embs_share_weight,
            d_k=model_opt.d_k,
            d_v=model_opt.d_v,
            d_model=model_opt.d_model,
            d_word_vec=model_opt.d_word_vec,
            d_inner=model_opt.d_inner_hid,
            n_layers=model_opt.n_layers,
            n_head=model_opt.n_head,
            dropout=model_opt.dropout)

        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')

        model.word_prob_prj = nn.LogSoftmax(dim=1)

        model = model.to(self.device)

        self.model = model
        self.model.eval()

        self.n_voca = model_opt.src_vocab_size

    def translate_batch(self, src_seq, src_pos):
        """ Translation work in one batch """

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            """ Indicate the position of an instance in a tensor. """
            return {inst_idx: tensor_position for tensor_position, inst_idx in
                    enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx,
                                n_prev_active_inst, n_bm):
            """ Collect tensor parts associated to active instances. """

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(
                src_seq,
                src_enc,
                inst_idx_to_position_map,
                active_inst_idx_list):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k]
                               for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_src_seq = collect_active_part(src_seq,
                                                 active_inst_idx,
                                                 n_prev_active_inst,
                                                 n_bm)
            active_src_enc = collect_active_part(src_enc,
                                                 active_inst_idx,
                                                 n_prev_active_inst,
                                                 n_bm)
            active_inst_idx_to_position_map = \
                get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(
                inst_dec_beams,
                len_dec_seq,
                src_seq,
                enc_output,
                inst_idx_to_position_map,
                n_bm):

            """ Decode and update beam status, and then return active beam idx
            """

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
                dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=self.device)
                dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
                return dec_partial_pos

            def predict_word(dec_seq,
                             dec_pos,
                             src_seq,
                             enc_output,
                             n_active_inst,
                             n_bm):
                ##############################################################
                # Make Mask
                # Find The Token Appeared in src sequence
                uniques = torch.unique(src_seq,
                                       dim=1).cpu().detach().numpy()
                # print(uniques.shape)

                p_gen_mask_shape = (src_seq.shape[0], self.n_voca)
                p_gen_mask = torch.tensor(np.zeros(p_gen_mask_shape),
                                          dtype=torch.float)

                batch_size = src_seq.shape[0]
                p_gen_mask[np.arange(batch_size)[:, None], uniques] = 1
                p_gen_mask = p_gen_mask.to(self.device)
                # print(p_gen_mask.shape)
                ############################################################

                dec_output, *_ = self.model.decoder(dec_seq,
                                                    dec_pos,
                                                    src_seq,
                                                    enc_output)
                print("dec_output.shape | before reshape", dec_output.shape)
                p_gen, *_ = self.model.p_generator(dec_seq,
                                                   dec_pos,
                                                   src_seq,
                                                   enc_output)
                p_gen = p_gen[:, -1, :]  # to get just last one.. why?
                # print("p_gen", p_gen.shape)

                p_gen = self.model.p_gen_linear(p_gen)
                p_gen = self.model.p_gen_sig(p_gen)

                dec_output = dec_output[:, -1, :]
                print("dec_output.shape | after reshape", dec_output.shape)

                seq_logit = self.model.tgt_word_prj(dec_output)  #
                print("seq_logit.shape | wo rd_prj", seq_logit.shape)

                seq_max_len = 1
                print("seq_max_len", seq_max_len)

                p_gen_mask = p_gen_mask[:, None, :]
                p_gen_mask = p_gen_mask[:, -1, :]

                print("p_gen_mask", p_gen_mask.shape)

                p_gen_mask = torch.repeat_interleave(p_gen_mask,
                                                     seq_max_len,
                                                     dim=1)
                masked_seq_logit = seq_logit * p_gen_mask
                print("masked_seq_logit.shape", masked_seq_logit.shape)
                ###########################################################
                softmax = torch.nn.Softmax(dim=1)
                prb_gen = softmax(seq_logit)
                prb_cp = softmax(masked_seq_logit)

                exclusive_copy_or_gen = True
                if exclusive_copy_or_gen:
                    p_gen = p_gen > 1/2
                    p_gen = p_gen.to(torch.float)

                prb = prb_gen * p_gen + prb_cp * (1 - p_gen)

                word_prob = prb.log()

                # Pick the last step: (bh * bm) * d_h

                # word_prob = F.log_softmax(self.model.tgt_word_prj(dec_output),
                #                           dim=1)

                word_prob = word_prob.view(n_active_inst, n_bm, -1)

                return word_prob

            def collect_active_inst_idx_list(inst_beams,
                                             word_prob,
                                             inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(
                        word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)  # int

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            dec_pos = prepare_beam_dec_pos(len_dec_seq,
                                           n_active_inst,
                                           n_bm)
            word_prob = predict_word(dec_seq,
                                     dec_pos,
                                     src_seq,
                                     enc_output,
                                     n_active_inst,  # what is n_active_inst?
                                     n_bm)  # what is n_bm?

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            # -- Encode
            src_seq, src_pos = src_seq.to(self.device), src_pos.to(self.device)
            src_enc, *_ = self.model.encoder(src_seq, src_pos)

            # -- Repeat data for beam search
            n_bm = self.opt.beam_size
            n_inst, len_s, d_h = src_enc.size()
            src_seq = src_seq.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)

            # -- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=self.device)
                              for _ in range(n_inst)]

            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(
                active_inst_idx_list)

            # -- Decode
#            seq_logit_group = []
            for len_dec_seq in range(1, self.model_opt.max_token_seq_len + 1):

                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams,
                    len_dec_seq,
                    src_seq,
                    src_enc,
                    inst_idx_to_position_map,
                    n_bm)
                # seq_logit_group.append(seq_logit)
                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                src_seq, src_enc, inst_idx_to_position_map = \
                    collate_active_info(src_seq,
                                        src_enc,
                                        inst_idx_to_position_map,
                                        active_inst_idx_list)

        batch_hyp, batch_scores = collect_hypothesis_and_scores(
            inst_dec_beams, self.opt.n_best)

        return batch_hyp, batch_scores #, seq_logit_group

