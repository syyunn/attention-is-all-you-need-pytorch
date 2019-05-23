''' Translate input text with trained model. '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm

import numpy as np
from dataset import collate_fn_translate, paired_collate_fn, TranslationDataset
from transformer.Translator import Translator
import transformer.Constants as Constants
import torch.nn.functional as F


from preprocess import read_instances_from_file, convert_instance_to_idx_seq


def cal_performance(pred, gold):
    """ Apply label smoothing if needed """

#    loss = cal_loss(pred, gold, smoothing)

#    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return n_correct

# Loss Currently Deprecated.
# def cal_loss(pred, gold, smoothing):
#     """ Calculate cross entropy loss, apply label smoothing if needed. """
#
#     gold = gold.contiguous().view(-1)
#
#     if smoothing:
#         eps = 0.1
#         n_class = pred.size(1)
#
#         one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
#         one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
#         log_prb = F.log_softmax(pred, dim=1)
#
#         non_pad_mask = gold.ne(Constants.PAD)
#         loss = -(one_hot * log_prb).sum(dim=1)
#         loss = loss.masked_select(non_pad_mask).sum()  # average later
#     else:
#         loss = F.cross_entropy(pred, gold,
#                                ignore_index=Constants.PAD,
#                                reduction='sum')
#
#     return loss


def main():
    """Main Function"""

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-src', required=True,
                        help='Source sequence to decode '
                             '(one line per sequence)')
    parser.add_argument('-tgt', required=True,
                        help='Target sequence to decode '
                             '(one line per sequence)')
    parser.add_argument('-vocab', required=True,
                        help='Source sequence to decode '
                             '(one line per sequence)')
    parser.add_argument('-log', default='translate_log.txt',
                        help="""Path to log the translation(test_inference) 
                        loss""")
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=30,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # Prepare DataLoader
    preprocess_data = torch.load(opt.vocab)
    preprocess_settings = preprocess_data['settings']
    test_src_word_insts = read_instances_from_file(
        opt.src,
        preprocess_settings.max_word_seq_len,
        preprocess_settings.keep_case)
    test_tgt_word_insts = read_instances_from_file(
        opt.tgt,
        preprocess_settings.max_word_seq_len,
        preprocess_settings.keep_case)
    test_src_insts = convert_instance_to_idx_seq(
        test_src_word_insts, preprocess_data['dict']['src'])
    test_tgt_insts = convert_instance_to_idx_seq(
        test_tgt_word_insts, preprocess_data['dict']['tgt'])

    test_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=preprocess_data['dict']['src'],
            tgt_word2idx=preprocess_data['dict']['tgt'],
            src_insts=test_src_insts,
            tgt_insts=test_tgt_insts),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)

    translator = Translator(opt)

    n_word_total = 0
    n_word_correct = 0

    with open(opt.output, 'w') as f:
        for batch in tqdm(test_loader,
                          mininterval=2,
                          desc='  - (Test)',
                          leave=False):
            # all_hyp, all_scores = translator.translate_batch(*batch)
            all_hyp, all_scores = translator.translate_batch(
                batch[0], batch[1])

            # print(all_hyp)
            # print(all_hyp[0])
            # print(len(all_hyp[0]))

            # pad with 0's fit to max_len in insts_group
            src_seqs = batch[0]
            # print(src_seqs.shape)
            tgt_seqs = batch[2]
            # print(tgt_seqs.shape)
            gold = tgt_seqs[:, 1:]
            # print(gold.shape)
            max_len = gold.shape[1]

            pred_seq = []
            for item in all_hyp:
                curr_item = item[0]
                curr_len = len(curr_item)
                # print(curr_len, max_len)
                # print(curr_len)
                if curr_len < max_len:
                    diff = max_len - curr_len
                    curr_item.extend([0]*diff)
                else:  # TODO: why does this case happen?
                    curr_item = curr_item[:max_len]
                pred_seq.append(curr_item)
            pred_seq = torch.LongTensor(np.array(pred_seq))
            pred_seq = pred_seq.view(opt.batch_size * max_len)

            n_correct = cal_performance(pred_seq, gold)

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

            # trs_log = "transformer_loss: {} |".format(trs_loss)
            #
            # with open(opt.log, 'a') as log_tf:
            #     log_tf.write(trs_log + '\n')

            count = 0
            for pred_seqs in all_hyp:
                src_seq = src_seqs[count]
                tgt_seq = tgt_seqs[count]
                for pred_seq in pred_seqs:
                    src_line = ' '.join([test_loader.dataset.src_idx2word[idx]
                                         for idx in src_seq.data.cpu().numpy()])
                    tgt_line = ' '.join([test_loader.dataset.tgt_idx2word[idx]
                                         for idx in tgt_seq.data.cpu().numpy()])
                    pred_line = ' '.join([test_loader.dataset.tgt_idx2word[idx]
                                          for idx in pred_seq])
                    f.write("\n ----------------------------------------------------------------------------------------------------------------------------------------------  \n")
                    f.write("\n [src]  " + src_line + '\n')
                    f.write("\n [tgt]  " + tgt_line + '\n')
                    f.write("\n [pred] " + pred_line + '\n')

                    count += 1

        accuracy = n_word_correct / n_word_total
        accr_log = "accuracy: {} |".format(accuracy)
        # print(accr_log)

        with open(opt.log, 'a') as log_tf:
            log_tf.write(accr_log + '\n')

    print('[Info] Finished.')


if __name__ == "__main__":
    main()
