import collections
import math
import os.path

from transformers import BartForConditionalGeneration as BartForConditionalGeneration_transformers
from bart import BartForConditionalGeneration
from tqdm import tqdm, trange
from torch.utils.data import Dataset
import torch.nn as nn
import torch
import numpy as np
import random
import pickle

def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.
    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return bleu, precisions, bp, ratio, translation_length, reference_length


class ModelForSent(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model = BartForConditionalGeneration_transformers(config)

    def forward(self, input_ids, attention_mask, decoder_input_ids=None, decoder_attention_mask=None, label_ids=None):

        if label_ids is None:
            return self.model.generate(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       max_length=150,
                                       num_beams=5,
                                       num_return_sequences=1,
                                       bos_token_id=1,
                                       eos_token_id=2,
                                       pad_token_id=0,
                                       )

        output = self.model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels=label_ids)

        return output[0]


class ModelForDoc(nn.Module):
    def __init__(self, config, rnn_idx=None):
        super().__init__()

        self.model = BartForConditionalGeneration(config)

        self.rnn_idx = rnn_idx
        self.use_context = True if max(rnn_idx) < 6 else False

    def forward(self, input_ids, attention_mask, decoder_input_ids=None, decoder_attention_mask=None, label_ids=None, print_path=None, return_dict_in_generate=False):

        if label_ids is None:
            if decoder_input_ids is not None:
                self.model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, return_dict=True, print_path=print_path)
                return
            return self.model.generate(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       max_length=150,
                                       num_beams=5,
                                       num_return_sequences=1,
                                       bos_token_id=1,
                                       eos_token_id=2,
                                       pad_token_id=0,
                                       output_scores=return_dict_in_generate,
                                       return_dict_in_generate=return_dict_in_generate,
                                       use_cache=False
                                       )

        output = self.model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels=label_ids)

        return output[0]

    def reset_context_state(self):
        for idx in self.rnn_idx:
            self.model.model.encoder.layers[idx].rnn.reset_query()
            self.model.model.decoder.layers[idx].rnn.reset_query()

    def set_decode(self, boolean):
        for idx in self.rnn_idx:
            self.model.model.encoder.layers[idx].decode = boolean
            self.model.model.decoder.layers[idx].decode = boolean


class MTDataset(Dataset):
    def __init__(self, ds):
        self.dataset = ds

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def process_data_for_sentences(vocab_to_ids, filename_src, filename_tgt):

    if os.path.exists("{}-cache".format(filename_src)):
        with open("{}-cache".format(filename_src), "rb") as fw:
            datasets = pickle.load(fw)

        return datasets

    all_input_ids = []

    with open(filename_src, 'r') as fr_src, open(filename_tgt, 'r') as fr_tgt:
        src_lines = fr_src.readlines()
        tgt_lines = fr_tgt.readlines()
        for i in trange(len(src_lines)):

            src_line = src_lines[i][:-1]
            tgt_line = tgt_lines[i][:-1]

            src_split = src_line.split('</s>')[:-1]
            tgt_split = tgt_line.split('</s>')[:-1]

            assert len(src_split) == len(tgt_split)

            for src, tgt in zip(src_split, tgt_split):

                encoder_tokens = src.split()
                decoder_tokens = tgt.split()

                encoder_tokens = ["<s>"] + encoder_tokens[:] + ["</s>"]
                decoder_tokens = ["<s>"] + decoder_tokens[:] + ["</s>"]

                encoder_input_ids = []
                decoder_input_ids = []

                for token in encoder_tokens:
                    if token not in vocab_to_ids:
                        print(token)
                        continue
                    encoder_input_ids.append(vocab_to_ids[token])
                for token in decoder_tokens:
                    if token not in vocab_to_ids:
                        print(token)
                        continue
                    decoder_input_ids.append(vocab_to_ids[token])

                all_input_ids.append(encoder_input_ids + [-100] + decoder_input_ids)
                # all_decoder_input_ids.append(decoder_input_ids)

    sorted_input_ids = sorted(all_input_ids, key=len)
    sorted_encoder_input_ids = []
    sorted_decoder_input_ids = []
    for _input_ids in sorted_input_ids:
        index = _input_ids.index(-100)
        sorted_encoder_input_ids.append(_input_ids[:index])
        sorted_decoder_input_ids.append(_input_ids[index+1:])

    datasets = []

    max_num_tokens = 4096
    cur_dataset = []
    max_length = 0

    for i in range(len(sorted_encoder_input_ids)):
        cur_encoder_input_ids = sorted_encoder_input_ids[i]
        cur_decoder_input_ids = sorted_decoder_input_ids[i]

        if (max(len(cur_decoder_input_ids),len(cur_encoder_input_ids)) * (len(cur_dataset) + 1)) > max_num_tokens:
            tmp_dataset = []
            for (encoder_ids, decoder_ids) in cur_dataset:
                tmp_encoder_ids = encoder_ids[:]
                tmp_decoder_ids = decoder_ids[:-1]
                tmp_label_ids = decoder_ids[1:]
                tmp_encoder_attention_mask = [1] * len(tmp_encoder_ids)
                tmp_decoder_attention_mask = [1] * len(tmp_decoder_ids)

                while len(tmp_encoder_ids) < max_length:
                    tmp_encoder_ids.append(0)
                    tmp_encoder_attention_mask.append(0)

                while len(tmp_decoder_ids) < max_length:
                    tmp_decoder_ids.append(0)
                    tmp_label_ids.append(-100)
                    tmp_decoder_attention_mask.append(0)
                tmp_dataset.append([tmp_encoder_ids, tmp_encoder_attention_mask, tmp_decoder_ids, tmp_decoder_attention_mask, tmp_label_ids])
            datasets.append(tmp_dataset[:])
            cur_dataset = []
            # num_tokens = 0
            max_length = 0
        # num_tokens = len(cur_decoder_input_ids) * (len(cur_dataset) + 1)
        cur_dataset.append((cur_encoder_input_ids, cur_decoder_input_ids))
        max_length = max([max_length, len(cur_encoder_input_ids), len(cur_decoder_input_ids)])

    tmp_dataset = []
    for (encoder_ids, decoder_ids) in cur_dataset:
        tmp_encoder_ids = encoder_ids[:]
        tmp_decoder_ids = decoder_ids[:-1]
        tmp_label_ids = decoder_ids[1:]
        tmp_encoder_attention_mask = [1] * len(tmp_encoder_ids)
        tmp_decoder_attention_mask = [1] * len(tmp_decoder_ids)

        while len(tmp_encoder_ids) < max_length:
            tmp_encoder_ids.append(0)
            tmp_encoder_attention_mask.append(0)

        while len(tmp_decoder_ids) < max_length:
            tmp_decoder_ids.append(0)
            tmp_label_ids.append(-100)
            tmp_decoder_attention_mask.append(0)
        tmp_dataset.append(
            [tmp_encoder_ids, tmp_encoder_attention_mask, tmp_decoder_ids, tmp_decoder_attention_mask, tmp_label_ids])
    if len(tmp_dataset) > 0:
        datasets.append(tmp_dataset[:])

    with open("{}-cache".format(filename_src), "wb") as fw:
        pickle.dump(datasets, fw)

    return datasets


def process_data_with_order(vocab_to_ids, filename_src, filename_tgt):
    max_length = 128

    docs = []
    with open(filename_src, 'r',encoding="utf-8") as fr_src, open(filename_tgt, 'r',encoding="utf-8") as fr_tgt:
        src_lines = fr_src.readlines()
        tgt_lines = fr_tgt.readlines()
        for i in trange(len(src_lines)):

            src_line = src_lines[i][:-1]
            tgt_line = tgt_lines[i][:-1]

            src_split = src_line.split('</s>')[:-1]
            tgt_split = tgt_line.split('</s>')[:-1]

            assert len(src_split) == len(tgt_split)

            doc = []
            for src, tgt in zip(src_split, tgt_split):

                encoder_tokens = src.split()
                decoder_tokens = tgt.split()

                if len(encoder_tokens) > (max_length - 2):
                    encoder_tokens = encoder_tokens[:max_length - 2]

                if len(decoder_tokens) > (max_length - 2):
                    decoder_tokens = decoder_tokens[:max_length - 2]

                encoder_tokens = ["<s>"] + encoder_tokens[:] + ["</s>"]
                decoder_tokens = ["<s>"] + decoder_tokens[:] + ["</s>"]

                encoder_input_ids = []
                decoder_input_ids = []

                for token in encoder_tokens:
                    if token not in vocab_to_ids:
                        print(token)
                        continue
                    encoder_input_ids.append(vocab_to_ids[token])
                for token in decoder_tokens:
                    if token not in vocab_to_ids:
                        print(token)
                        continue
                    decoder_input_ids.append(vocab_to_ids[token])

                attention_mask = [1 for _ in range(len(encoder_input_ids))]
                label_ids = decoder_input_ids[1:]
                decoder_input_ids = decoder_input_ids[:-1]
                decoder_attention_mask = [1 for _ in range(len(decoder_input_ids))]

                while len(encoder_input_ids) < max_length:
                    encoder_input_ids.append(0)
                    attention_mask.append(0)

                while len(decoder_input_ids) < max_length:
                    decoder_input_ids.append(0)
                    decoder_attention_mask.append(0)
                    label_ids.append(-100)

                doc.append([np.array(encoder_input_ids),
                            np.array(attention_mask),
                            np.array(decoder_input_ids),
                            np.array(decoder_attention_mask),
                            np.array(label_ids)])
            docs.append(doc)

    return docs


def shuffle(docs, batch_size):
    partitions = 3
    random.shuffle(docs)
    length = len(docs) // partitions

    shuffled_docs = []
    for i in range(partitions):
        if i == partitions - 1:
            shuffled_subdocs = _shuffle(docs[length * i:], batch_size)
        else:
            shuffled_subdocs = _shuffle(docs[length * i:length * (i + 1)], batch_size)

        shuffled_docs = shuffled_docs[:] + shuffled_subdocs[:]
    return shuffled_docs


def _shuffle(docs, batch_size):
    max_length = 128

    doc_pad = [[0 for _ in range(max_length)] for _ in range(4)] + [[-100 for _ in range(max_length)]]
    docs.sort(key=len)

    new_docs = []
    doc_batch = [docs[0][:]]
    doc_length = len(docs[0][:])

    all = 0
    for i, doc in enumerate(docs[1:]):
        if len(doc) <= (doc_length + 10) and len(doc_batch) < batch_size:
            doc_batch.append(doc[:])
        else:
            doc_batch_with_pad = []
            max_doc_length = len(doc_batch[-1])
            for d_i in range(len(doc_batch)):
                d = doc_batch[d_i]
                while len(d) < max_doc_length:
                    d.append(doc_pad[:])
                doc_batch_with_pad.append(d)
            new_docs.append(np.array(doc_batch_with_pad))
            doc_batch = [doc[:]]
            doc_length = len(doc)
        all += len(doc)

    doc_batch_with_pad = []
    max_doc_length = len(doc_batch[-1])
    for d_i in range(len(doc_batch)):
        d = doc_batch[d_i]
        while len(d) < max_doc_length:
            d.append(doc_pad[:])
        doc_batch_with_pad.append(d)
    new_docs.append(np.array(doc_batch_with_pad))

    steps = 0
    for d in new_docs:
        steps += len(d)

    return new_docs


def evaluate_for_sentences(model, dataloader, ids_to_vocab, output_file, wt_context=False):
    eos_id = 2

    refs = []
    hyps = []

    for _batch in tqdm(dataloader):
        batch = torch.Tensor(_batch).cuda().long()
        input_ids = batch[:, 0]  # .cuda().long()
        attention_mask = batch[:, 1]  # .cuda().long()
        label_ids = batch[:, 4].tolist()  # .cuda().long()

        output = model(input_ids, attention_mask)

        output = output.cpu().tolist()
        for i in range(len(output)):
            pred = output[i]
            index = pred.index(eos_id) if eos_id in pred else -1
            pred_tokens = []
            for t in pred[1:index]:
                pred_tokens.append(ids_to_vocab[t])
            pred_str = " ".join(pred_tokens)
            pred_str = pred_str.replace("@@ ", "")
            hyps.append(pred_str.split())

            target = label_ids[i]
            index = target.index(eos_id)
            target_tokens = []
            for t in target[:index]:
                target_tokens.append(ids_to_vocab[t])
            target_str = " ".join(target_tokens)
            target_str = target_str.replace("@@ ", "")
            refs.append([target_str.split()])

    score = compute_bleu(refs, hyps, max_order=4)[0]
    with open(output_file, 'a') as fw:
        fw.write("{} context, BLEU Score: {}\n".format("with" if wt_context else "no", score))
    print("{} context, BLEU Score: {}\n".format("with" if wt_context else "no", score))

    return score


def evaluate_with_order(model, dataloader, ids_to_vocab, output_file, shuffled, update=True, reset=True):
    use_context = model.use_context

    eos_id = 2

    refs = []
    hyps = []
    doc_refs = []
    doc_hyps = []

    for doc in tqdm(dataloader):
        doc_batch = doc[0]
        doc_batch = torch.transpose(doc_batch, 0, 1)
        doc_batch = torch.transpose(doc_batch, 1, 2)

        if shuffled:
            doc_batch = doc_batch[torch.randperm(doc_batch.size()[0])]
        doc_batch = doc_batch.numpy()

        if use_context and reset:
            model.reset_context_state()
        doc_ref_str = ""
        doc_hyp_str = ""
        batch_index = 1
        for batch in (doc_batch):
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, label_ids = batch
            input_ids = torch.Tensor(input_ids).cuda().long()
            attention_mask = torch.Tensor(attention_mask).cuda().long()
            label_ids = label_ids.tolist()

            if use_context:
                model.set_decode(True)

            output = model(input_ids, attention_mask)

            if use_context and update:
                model.set_decode(False)
                decoder_attention_mask = torch.ones(output.size()).cuda()
                model(input_ids, attention_mask, output, decoder_attention_mask)

            output = output.cpu().tolist()

            batch_index += 1

            for i in range(len(output)):
                pred = output[i]
                index = pred.index(eos_id) if eos_id in pred else -1
                pred_tokens = []
                for t in pred[1:index]:
                    pred_tokens.append(ids_to_vocab[t])
                pred_str = " ".join(pred_tokens)
                pred_str = pred_str.replace("@@ ", "")

                target = label_ids[i]
                index = target.index(eos_id)
                target_tokens = []
                for t in target[:index]:
                    target_tokens.append(ids_to_vocab[t])
                target_str = " ".join(target_tokens)
                target_str = target_str.replace("@@ ", "")

                doc_ref_str = doc_ref_str + target_str + " "
                doc_hyp_str = doc_hyp_str + pred_str + " "
                refs.append([target_str.split()])
                hyps.append(pred_str.split())
        # return
        doc_refs.append([doc_ref_str.split()])
        doc_hyps.append(doc_hyp_str.split())

    with open(output_file, 'a') as fw:
        score_sent = "sent BLEU Score: {}".format(compute_bleu(refs, hyps, max_order=4)[0])
        score_doc = "doc BLEU Score: {}".format(compute_bleu(doc_refs, doc_hyps, max_order=4)[0])
        fw.write(score_sent + '\n')
        fw.write(score_doc + '\n')
    print(score_sent + '\n')
    print(score_doc + '\n')


class ReverseSqrtScheduler:
    def __init__(self, optimizer, lr, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

        self.decay_factor = [_lr * n_warmup_steps ** 0.5 for _lr in lr]
        self.lr_step = [(_lr - 0) / n_warmup_steps for _lr in lr]

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _update_learning_rate(self):
        self.n_steps += 1
        if self.n_steps < self.n_warmup_steps:
            lr = [self.n_steps * _lr for _lr in self.lr_step]
        else:
            lr = [_decay_factor * self.n_steps ** -0.5 for _decay_factor in self.decay_factor]

        for i, param_group in enumerate(self._optimizer.param_groups):
            param_group['lr'] = lr[i]
