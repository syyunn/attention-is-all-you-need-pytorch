""" Define the Transformer model """
import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang"
__editor__ = "Zachary Yoon"


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i
                               in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q,
                                                    -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s),
                   device=seq.device,
                   dtype=torch.uint8),
        diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1,
                                                          -1)  # b x ls x ls

    return subsequent_mask


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            n_src_vocab,
            len_max_seq,
            d_word_vec,
            n_layers,
            n_head,
            d_k,
            d_v,
            d_model,
            d_inner,
            dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1
        # let model to figure out how many positions to prepare for
        # Positional Encoding

        self.src_word_emb = nn.Embedding(
            n_src_vocab,
            d_word_vec,
            padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(
                n_position,
                d_word_vec,
                padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        # since this is the self-attention, both seq_k and  seq_q gets a
        # src_input
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    """ A decoder model with self attention mechanism. """

    def __init__(
            self,
            n_tgt_vocab,
            len_max_seq,
            d_word_vec,
            n_layers,
            n_head,
            d_k,
            d_v,
            d_model,
            d_inner,
            dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab,
            d_word_vec,
            padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position,
                                        d_word_vec,
                                        padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model,
                         d_inner,
                         n_head,
                         d_k,
                         d_v,
                         dropout=dropout)
            for _ in range(n_layers)])

    def forward(self,
                tgt_seq,
                tgt_pos,
                src_seq,
                enc_output,
                return_attns=False):
        # print("tgt_seq | tgt_seq.   shape", tgt_seq.shape)

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq,
                                                     seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq,
                                                  seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)
        # print("before multi-heads | dec_output.shape", dec_output.shape)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output,
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)
            # print("dec_output_loop.shape", dec_output.shape)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class PGEN(nn.Module):
    """ A decoder model with self attention mechanism. """

    def __init__(
            self,
            n_tgt_vocab,
            len_max_seq,
            d_word_vec,
            n_layers,
            n_head,
            d_k,
            d_v,
            d_model,
            d_inner,
            dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab,
            d_word_vec,
            padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position,
                                        d_word_vec,
                                        padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model,
                         d_inner,
                         n_head,
                         d_k,
                         d_v,
                         dropout=dropout)
            for _ in range(n_layers)])

    def forward(self,
                tgt_seq,
                tgt_pos,
                src_seq,
                enc_output,
                return_attns=False):
        # print("tgt_seq | tgt_seq.   shape", tgt_seq.shape)

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq,
                                                     seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq,
                                                  seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)
        # print("before multi-heads | dec_output.shape", dec_output.shape)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output,
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)
            # print("dec_output_loop.shape", dec_output.shape)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            n_src_vocab,
            n_tgt_vocab,
            len_max_seq,
            device,
            d_word_vec=512,
            d_model=512,
            d_inner=2048,
            n_layers=6,
            n_head=8,
            d_k=64,
            d_v=64,
            dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True,
            allow_copy=True):

        super().__init__()
        self.n_src_vocab = n_src_vocab
        self.n_tgt_vocab = n_tgt_vocab
        self.allow_copy = allow_copy
        self.device = device

        self.tgt_emb_prj_weight_sharing = tgt_emb_prj_weight_sharing
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab,
            len_max_seq=len_max_seq,
            d_word_vec=d_word_vec,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab,
            len_max_seq=len_max_seq,
            d_word_vec=d_word_vec,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout)

        self.p_generator = PGEN(
            n_tgt_vocab=n_tgt_vocab,
            len_max_seq=len_max_seq,
            d_word_vec=d_word_vec,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)
        # this makes the tensor to be enlarged to the size of n_voca

        # self.final_linear = nn.Linear(n_tgt_vocab, n_tgt_vocab, bias=True)
        # nn.init.xavier_normal_(self.final_linear.weight)
        # self.final_relu = nn.ReLU()

        # - Added by Zachary | p_gen generator ##################
        if allow_copy:
            print("Copying Mechanism Initialized")
            self.p_gen_linear = nn.Linear(d_model, 1, bias=False)
            self.p_gen_sig = nn.Sigmoid()
            nn.init.xavier_normal_(self.p_gen_linear.weight)
        #########################################################

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the
            # final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
                "To share word embedding table, the vocabulary size of " \
                "src/tgt shall be the same. "
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self,
                src_seq,
                src_pos,
                tgt_seq,
                tgt_pos):

        # - Added by Zachary ##################################################
        if self.allow_copy:
            # print("[info] copying mask generation..")
            # To add the copy mask that has 1 only in the position of src
            # sequence
            # To use this mechanism, following condition must be fulfilled
            assert self.n_src_vocab == self.n_tgt_vocab

            # Find The Token Appeared in src sequence
            uniques = torch.unique(src_seq, dim=1).cpu().detach().numpy()
            # print(uniques.shape)

            p_gen_mask_shape = (src_seq.shape[0], self.n_tgt_vocab)
            p_gen_mask = torch.tensor(np.zeros(p_gen_mask_shape),
                                      dtype=torch.float)

            batch_size = src_seq.shape[0]
            p_gen_mask[np.arange(batch_size)[:, None], uniques] = 1
            p_gen_mask = p_gen_mask.to(self.device)
            # print(p_gen_mask.shape)
        # - ##################################################################

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos)
        # print("enc_ouptut.shape", enc_output.shape)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        # what happens when enc_output goes into the decoder?
        # print("dec_output.shape", dec_output.shape)

        p_gen, *_ = self.p_generator(tgt_seq, tgt_pos, src_seq, enc_output)

        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale
        # print(seq_logit)
        # print("seq_logit.shape", seq_logit.shape)
        seq_max_len = seq_logit.shape[1]
        # print("seq_max_len", seq_max_len)

        # Extend dimension upto max sequence_length ###############
        p_gen_mask = p_gen_mask[:, None, :]
        p_gen_mask = torch.repeat_interleave(p_gen_mask, seq_max_len, dim=1)
        # print(p_gen_mask)
        # print(p_gen_mask.shape)
        ###########################################################

        # Added by Zachary | p_gen generator ##
        if self.allow_copy:
            p_gen = self.p_gen_linear(p_gen)
            p_gen = self.p_gen_sig(p_gen)

            masked_seq_logit = seq_logit * p_gen_mask

            # to penalize redundancy
            # to count duplicated prediction

###############################################################################
# redundancy loss unit
            # def _cal_redun(seq_logit):
            #     redun_score = 0
            #     seq_logit_redun = seq_logit.max(2)[1]
            #     for sentence in seq_logit_redun:
            #         sent_dict = dict()
            #         sentence_redun = 0
            #         for item in sentence:
            #             token_idx = int(item.data.cpu())
            #             if token_idx in sent_dict.keys():
            #                 # sent_dict[token_idx] += 1
            #                 sentence_redun += 1
            #             else:
            #                 sent_dict[token_idx] = 1
            #         redun_score += sentence_redun
            #     return redun_score

            # redun_seq = _cal_redun(seq_logit)
            # redun_masked = _cal_redun(masked_seq_logit)
            #
            # redun = redun_seq + redun_masked
###############################################################################

            # flatten
            seq_logit = seq_logit.view(-1, seq_logit.size(2))
            masked_seq_logit = masked_seq_logit.view(-1,
                                                     masked_seq_logit.size(2))
            p_gen = p_gen.view(-1, p_gen.size(2))



        ###########################################################

        # else:
        #     final_output = seq_logit.view(-1, seq_logit.size(2))
        #     # .view(-1, seq_logit.size(2)) is to turn batch-wise into
        #     # batch-aggregation.

        return seq_logit, masked_seq_logit, p_gen  # , redun
        # this returns tensor.shape = (batch_size * max_seq_len, n_voca)
