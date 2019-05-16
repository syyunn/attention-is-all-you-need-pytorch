''' Translate input text with trained model. '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm

from dataset import paired_collate_fn, TranslationDataset
from transformer.Translator import Translator
from preprocess import read_instances_from_file, convert_instance_to_idx_seq


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

    with open(opt.output, 'w') as f:
        for batch in tqdm(test_loader,
                          mininterval=2,
                          desc='  - (Test)',
                          leave=False):
            # all_hyp, all_scores = translator.translate_batch(*batch)
            all_hyp, all_scores = translator.translate_batch(batch[0],
                                                             batch[1])
            src_seqs = batch[0]
            tgt_seqs = batch[2]
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

    print('[Info] Finished.')


if __name__ == "__main__":
    main()
