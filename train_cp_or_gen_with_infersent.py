"""
This script handling the training process.
"""

import argparse
import math
import time

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import transformer.Constants as Constants
from dataset import TranslationDataset, paired_collate_fn
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

import load


def cal_performance(pred_gen,
                    pred_cp,
                    p_gen,
                    gold,
                    smoothing=False):
    """ Apply label smoothing if needed """
    # print("gold1", gold)
    # print(gold.shape)
    loss, pred = cal_loss_cp(pred_gen,
                             pred_cp,
                             p_gen,
                             gold)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct, pred


def cal_loss_cp(pred_gen,
                pred_cp,
                p_gen,
                gold):
    """ Calculate cross entropy loss, apply label smoothing if needed. """

    gold = gold.contiguous().view(-1)
    softmax = torch.nn.Softmax(dim=1)

    one_hot = torch.zeros_like(pred_gen).scatter(1, gold.view(-1, 1), 1)

    prb_gen = softmax(pred_gen)

    prb_cp = softmax(pred_cp)

    exclusive_cp_or_gen = True
    if exclusive_cp_or_gen:  # this works better for copying
        p_gen = p_gen > 1 / 2
        p_gen = p_gen.to(torch.float)
    prb = prb_gen * p_gen + prb_cp * (1 - p_gen)
    # print("prb", prb)
    # print("prb.shape", prb.shape)
    log_prb = prb.log()
    # print("log_prb", log_prb)

    non_pad_mask = gold.ne(Constants.PAD)

    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.masked_select(non_pad_mask).sum()  # average later

    return loss, log_prb


def train_epoch(infersent_model,
                log_train_file,
                model,
                training_data,
                optimizer,
                device,
                smoothing):
    """ Epoch operation in training phase """

    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    total_batch_num = len(training_data)
    print("total_batch_num: {}".format(total_batch_num))
    # count = 0
    for batch in tqdm(
            training_data,
            mininterval=2,
            desc='  - (Training)   ',
            leave=False):

        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:, 1:]

        # infersent
        batch_src_to_feed_infersent = []
        for seq in src_seq:
            src_line = ' '.join([training_data.dataset.src_idx2word[idx]
                                 for idx in seq.data.cpu().numpy()])
            src_line_clear = src_line[3:].split('</s>')[0]
            batch_src_to_feed_infersent.append(src_line_clear)

        batch_src_infersent_enc = infersent_model.encode(
            batch_src_to_feed_infersent)

        batch_size = batch_src_infersent_enc.shape[0]



        # forward
        optimizer.zero_grad()
        pred_gen, pred_cp, p_gen = model(src_seq,
                                         src_pos,
                                         tgt_seq,
                                         tgt_pos)

        # backward
        trs_loss, n_correct, pred = cal_performance(pred_gen,
                                                    pred_cp,
                                                    p_gen,
                                                    gold,
                                                    smoothing=smoothing)

        def _translate(torch_tokens):
            translation = ' '.join([training_data.dataset.tgt_idx2word[idx]
                                    for idx in torch_tokens.data.cpu().numpy()])
            translation = translation.split('<blank>')[0]
            translation = ' ' + translation
            return translation

        pred_max = pred.view(batch_size, -1)

        batch_pred_to_feed_infersent = []
        for sent_token in pred_max:
            translated_pred = _translate(sent_token)
            # print(translated_pred)
            batch_pred_to_feed_infersent.append(translated_pred)

        # batch_tgt_infersent_enc = infersent_model.encode(
        #     batch_tgt_to_feed_infersent)
        batch_pred_infersent_enc = infersent_model.encode(
            batch_pred_to_feed_infersent)

        sumrz_devit = batch_src_infersent_enc - batch_pred_infersent_enc

        general_permittance = 1.067753
        dists = np.linalg.norm(sumrz_devit, axis=1)

        dists_error = dists - general_permittance

        positivedx= np.where(dists_error > 0)[0]

        ifs_loss_multiplier = 16000

        ifs_loss = np.mean(dists_error[positivedx]) * ifs_loss_multiplier
        ifs_log = "infersent_loss: {} |".format(ifs_loss)
        print(ifs_log)

        trs_log = "trs_loss: {}".format(trs_loss)
        print(trs_log)

        final_loss = trs_loss + ifs_loss
        final_log = "total_loss : {}".format(final_loss)
        print(final_log)

        with open(log_train_file, 'a') as log_tf:
            # print('logging!')
            log_tf.write(trs_log + ifs_log + final_log + '\n')

        final_loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += final_loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def eval_epoch(infersent_model,
               log_valid_file,
               model,
               validation_data,
               device):
    """ Epoch operation in evaluation phase """

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data,
                mininterval=2,
                desc='  - (Validation) ',
                leave=False):

            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # infersent
            batch_src_to_feed_infersent = []
            for seq in src_seq:
                src_line = ' '.join([validation_data.dataset.src_idx2word[idx]
                                     for idx in seq.data.cpu().numpy()])
                src_line_clear = src_line[3:].split('</s>')[0]
                batch_src_to_feed_infersent.append(src_line_clear)

            batch_src_infersent_enc = infersent_model.encode(
                batch_src_to_feed_infersent)

            batch_size = batch_src_infersent_enc.shape[0]

            # forward
            pred_gen, pred_cp, p_gen = model(src_seq,
                                             src_pos,
                                             tgt_seq,
                                             tgt_pos)

            trs_loss, n_correct, pred = cal_performance(pred_gen,
                                                    pred_cp,
                                                    p_gen,
                                                    gold)

            def _translate(torch_tokens):
                translation = ' '.join([validation_data.dataset.tgt_idx2word[idx]
                                        for idx in
                                        torch_tokens.data.cpu().numpy()])
                translation = translation.split('<blank>')[0]
                translation = ' ' + translation
                return translation

            pred_max = pred.view(batch_size, -1)

            batch_pred_to_feed_infersent = []
            for sent_token in pred_max:
                translated_pred = _translate(sent_token)
                batch_pred_to_feed_infersent.append(translated_pred)

            batch_pred_infersent_enc = infersent_model.encode(
                batch_pred_to_feed_infersent)

            sumrz_devit = batch_src_infersent_enc - batch_pred_infersent_enc

            general_permittance = 1.067753
            dists = np.linalg.norm(sumrz_devit, axis=1)

            dists_error = dists - general_permittance

            positivedx = np.where(dists_error > 0)[0]

            ifs_loss_multiplier = 16000

            ifs_loss = np.mean(dists_error[positivedx]) * ifs_loss_multiplier
            ifs_log = "infersent_loss: {} |".format(ifs_loss)
            print(ifs_log)

            trs_log = "trs_loss: {}".format(trs_loss)
            print(trs_log)

            final_loss = trs_loss + ifs_loss
            final_log = "total_loss : {}".format(final_loss)
            print(final_log)

            with open(log_valid_file, 'a') as log_tf:
                # print('logging!')
                log_tf.write(trs_log + ifs_log + final_log + '\n')

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def train(infersent_model,
          model,
          training_data,
          validation_data,
          optimizer,
          device,
          opt):
    """ Start training """

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + 'train.log'
        log_valid_file = opt.log + 'valid.log'

        print('[Info] Training performance will be written to file: {} and '
              '{}'.format(log_train_file,
                          log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') \
                as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []

    def get_lr(optimizer):
        for param_group in optimizer._optimizer.param_groups:
            return param_group['lr']

    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')
        print('[ LR', get_lr(optimizer), ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            infersent_model,
            log_train_file,
            model,
            training_data,
            optimizer,
            device,
            smoothing=opt.label_smoothing)

        log = '  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60)

        print(log)

        with open(log_train_file, 'a') as log_tf:
            # print('logging!')
            log_tf.write(log + '\n')

        start = time.time()
        valid_loss, valid_accu = eval_epoch(infersent_model,
                                            log_valid_file,
                                            model,
                                            validation_data,
                                            device)
        log = '  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60)
        print(log)

        with open(log_valid_file, 'a') as log_vf:
            # print('logging!')
            log_vf.write(log + '\n')

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.\
                    format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.\
                    format(accu=100*valid_accu)
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))


def main():
    """ Main function """
    parser = argparse.ArgumentParser()

    parser.add_argument('-data',
                        default=
                        "/home/zachary/projects/"
                        "attention-is-all-you-need-pytorch/data/"
                        "gigaword.low.pt",
                        required=True)

    parser.add_argument('-epoch', type=int, default=7)
    parser.add_argument('-batch_size', type=int, default=64)

    # parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-allow_copy', action='store_false')

    parser.add_argument('-log',
                        default=None)
    parser.add_argument('-save_model',
                        default='trained')
    parser.add_argument('-save_mode',
                        type=str,
                        choices=['all', 'best'],
                        default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    # ========= Loading Dataset ========= #
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len  # default is 50

    training_data, validation_data = prepare_dataloaders(data, opt)

    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    # ========= Preparing Model ========= #
    if opt.embs_share_weight:
        assert training_data.dataset.src_word2idx == \
               training_data.dataset.tgt_word2idx, \
               'The src/tgt word2idx table are different but asked ' \
               'to share word embedding.'

    print(opt)

    device = torch.device('cuda' if opt.cuda else 'cpu')
    transformer = Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_token_seq_len,
        device=device,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        allow_copy=opt.allow_copy).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad,
                   transformer.parameters()),
            betas=(0.9, 0.98),
            eps=1e-09),
        opt.d_model,
        opt.n_warmup_steps)

    # optimizer = optim.Adam(filter(lambda x: x.requires_grad,
    #                               transformer.parameters()))
    infersent = load.infersent()

    train(infersent,
          transformer,
          training_data,
          validation_data,
          optimizer,
          device,
          opt)


def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader ========= #
    train_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['train']['src'],
            tgt_insts=data['train']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['valid']['src'],
            tgt_insts=data['valid']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)
    return train_loader, valid_loader


if __name__ == '__main__':
    main()
