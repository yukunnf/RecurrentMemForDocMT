from utils import *
import argparse
import os
from torch.utils.data import DataLoader
from transformers import BartConfig, AdamW

parser = argparse.ArgumentParser()
parser.add_argument('--training-mode', required=True, type=str,
                    help="finetune: finetune docMT from sentence Transformer, sentence: train sentence Transformer")
parser.add_argument("--dataset", required=True, type=str,
                    help="nc2016：News, europarl7：Europarl, iwslt17：TED, as demonstrated in paper")
parser.add_argument("--data-path", required=True, type=str,
                    help="path for checkpoint and processed data")
parser.add_argument("--output-path", required=True, type=str,
                    help="path for output files")
parser.add_argument("--batch-size", required=True, type=int)
parser.add_argument("--dropout", required=True, type=float)
parser.add_argument("--dropout-mem", required=True, type=float,
                    help="dropout for contextual memory related parameters")
parser.add_argument("--warmup-steps", required=True, type=int)
parser.add_argument("--min-steps", required=True, type=int)
parser.add_argument("--log-steps", required=True, type=int)
parser.add_argument("--eval-steps", required=True, type=int)

parser.add_argument("--learning-rate-sentence", required=True, type=float,
                    help="learning-rate for the whole model except contextual realated parameter")
parser.add_argument("--learning-rate-finetune", required=True, type=float,
                    help="learning-rate only for contextual memory related parameter ")
parser.add_argument('--mem-index', required=True, type=str,
                    help="layer index to enable contextual memory, 0: enable only 1st layer, 345: enable last 3 layers"
                         "6 for disable memory")
parser.add_argument("--mem-length", required=True, type=int,
                    help="memory allocation size")
parser.add_argument("--mem-side", required=True, type=str,
                    help="encoder, decoder, both")
parser.add_argument("--mem-pos", required=True, type=str,
                    help="sinusoidal: add positional encoding onto the contextual memory vector")
parser.add_argument("--min-optimize-step", required=True, type=int,
                    help="the lower bound of group optimization as stated in section 5.3")
parser.add_argument("--max-optimize-step", required=True, type=int,
                    help="the upper bound of group optimization as stated in section 5.3")

args = parser.parse_args()
print(args)

TRAINING_MODE = args.training_mode
DSNAME = args.dataset
OUTPUT_FILE = args.output_path
PATH_PREFIX = args.data_path

MEMORY_POS = args.mem_pos
MEMORY_LAYERS = [int(c) for c in args.mem_index]
MEMORY_LENGTH = args.mem_length
MEMORY_SIDE = args.mem_side

TRAIN_BSZ = args.batch_size
DROPOUT = args.dropout
MEMORY_DROPOUT = args.dropout_mem
WARMUP_STEPS = args.warmup_steps
MIN_STEP = args.min_steps
LOG_STEP = args.log_steps
EVAL_STEP = args.eval_steps

LR_PRETRAIN = args.learning_rate_sentence
LR_FINETUNE = args.learning_rate_finetune

MIN_OPTIMIZE_WINDOW_SIZE = args.min_optimize_step
MAX_OPTIMIZE_WINDOW_SIZE = args.max_optimize_step
OPTIMIZE_WINDOW_SIZE = [_ for _ in range(MIN_OPTIMIZE_WINDOW_SIZE, MAX_OPTIMIZE_WINDOW_SIZE + 1)]

with open(OUTPUT_FILE, 'a') as fw:
    fw.write("Arguments: {} \n".format(args))


def main():

    ckpt_prefix = "{}/ckpt".format(PATH_PREFIX)
    data_prefix = "{}/{}.tokenized.en-de".format(PATH_PREFIX, DSNAME)
    vocab_path = "{}/{}-vocab.txt".format(PATH_PREFIX, DSNAME)

    vocab_to_ids = {}
    ids_to_vocab = {}
    with open(vocab_path, 'r', encoding="utf-8") as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines):
            v = line[:-1]
            vocab_to_ids[v] = i
            ids_to_vocab[i] = v

    if TRAINING_MODE == "sentence":
        train_dataset = process_data_for_sentences(vocab_to_ids, "{}/train.en".format(data_prefix),
                                                   "{}/train.de".format(data_prefix))
        train_dataloader = DataLoader(dataset=MTDataset(train_dataset), pin_memory=True, batch_size=1,
                                      shuffle=True)
        test_dataset = process_data_for_sentences(vocab_to_ids, "{}/valid.en".format(data_prefix),
                                                  "{}/valid.de".format(data_prefix))
        test_dataloader = DataLoader(dataset=MTDataset(test_dataset), pin_memory=True, batch_size=1)

        config = BartConfig(vocab_size=len(vocab_to_ids), d_model=512, encoder_ffn_dim=2048, decoder_ffn_dim=2048,
                            pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1, encoder_layers=6,
                            decoder_layers=6, encoder_attention_heads=8, decoder_attention_heads=8, dropout=DROPOUT,
                            attention_dropout=DROPOUT)
        model = ModelForSent(config)

    else:
        raw_train_dataset = process_data_with_order(vocab_to_ids, "{}/train.en".format(data_prefix),
                                                    "{}/train.de".format(data_prefix))
        test_dataset = process_data_with_order(vocab_to_ids, "{}/valid.en".format(data_prefix),
                                               "{}/valid.de".format(data_prefix))
        test_dataset = shuffle(test_dataset, batch_size=1)
        test_dataloader = DataLoader(dataset=MTDataset(test_dataset), pin_memory=True, batch_size=1)

        config = BartConfig(vocab_size=len(vocab_to_ids), d_model=512, encoder_ffn_dim=2048, decoder_ffn_dim=2048,
                            pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1, encoder_layers=6,
                            decoder_layers=6, encoder_attention_heads=8, decoder_attention_heads=8, dropout=DROPOUT,
                            attention_dropout=DROPOUT, rnn_dropout=MEMORY_DROPOUT, rnn_query_length=MEMORY_LENGTH,
                            rnn_idx=MEMORY_LAYERS, rnn_side=MEMORY_SIDE, pos_type=MEMORY_POS)

        model = ModelForDoc(config, MEMORY_LAYERS)

    model.cuda()

    if TRAINING_MODE == "finetune":
        model.load_state_dict(torch.load('{}/ckpt-sentence-{}'.format(ckpt_prefix, DSNAME)), strict=False)
        model.reset_context_state()
        model.set_decode(False)

    pretrained_p = []
    fintune_p = []
    for (name, p) in model.named_parameters():
        if "rnn" in name:
            fintune_p.append(p)
        else:
            pretrained_p.append(p)

    optimizer = AdamW([{'params': pretrained_p, 'lr': LR_PRETRAIN}, {'params': fintune_p, 'lr': LR_FINETUNE}])
    scheduler = ReverseSqrtScheduler(optimizer, [LR_PRETRAIN, LR_FINETUNE], WARMUP_STEPS)

    update_step = 0

    log_loss_num = 0
    for epoch in range(100):

        model.train()
        epoch_loss = 0
        log_loss = 0
        num_batches = 0

        if TRAINING_MODE == "sentence":
            for _batch in tqdm(train_dataloader):
                batch = torch.Tensor(_batch).cuda().long()
                input_ids = batch[:, 0]  # .cuda().long()
                attention_mask = batch[:, 1]  # .cuda().long()
                decoder_input_ids = batch[:, 2]  # .cuda().long()
                decoder_attention_mask = batch[:, 3]  # .cuda().long()
                label_ids = batch[:, 4].contiguous()  # .cuda().long()

                loss = model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, label_ids)

                epoch_loss += loss.item()
                loss.backward()

                scheduler.step_and_update_lr()
                scheduler.zero_grad()
                num_batches += 1
                update_step += 1

                if update_step % LOG_STEP == 0:
                    with open(OUTPUT_FILE, 'a') as fw:
                        fw.write("Epoch {} loss: {}\n".format(epoch, epoch_loss / num_batches))
                    print("Epoch {} loss: {}\n".format(epoch, epoch_loss / num_batches))

                if update_step >= MIN_STEP and update_step % EVAL_STEP == 0:
                    with torch.no_grad():
                        model.eval()
                        score = evaluate_for_sentences(model, test_dataloader, ids_to_vocab, OUTPUT_FILE)
                    model.train()

        else:

            train_dataset = shuffle(raw_train_dataset, TRAIN_BSZ)
            train_dataloader = DataLoader(dataset=MTDataset(train_dataset), pin_memory=True, batch_size=1, shuffle=True)

            for doc in tqdm(train_dataloader):
                doc_batch = doc[0]

                doc_batch = torch.transpose(doc_batch, 0, 1)
                doc_batch = torch.transpose(doc_batch, 1, 2)

                doc_batch = doc_batch.numpy()

                model.reset_context_state()
                model.set_decode(False)

                loss_num = 0

                doc_length = doc_batch.shape[0]
                do_eval = False
                op_size = min(random.choice(OPTIMIZE_WINDOW_SIZE), doc_length)

                for batch in doc_batch:
                    input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, label_ids = batch

                    input_ids = torch.Tensor(input_ids).cuda().long()
                    attention_mask = torch.Tensor(attention_mask).cuda().long()
                    decoder_input_ids = torch.Tensor(decoder_input_ids).cuda().long()
                    decoder_attention_mask = torch.Tensor(decoder_attention_mask).cuda().long()
                    label_ids = torch.Tensor(label_ids).cuda().long()

                    loss = model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, label_ids)

                    loss = loss / op_size
                    log_loss += loss.item()
                    epoch_loss += loss.item()
                    loss.backward(retain_graph=True)

                    loss_num += 1
                    doc_length -= 1

                    if loss_num == op_size:
                        update_step += 1
                        num_batches += 1

                        scheduler.step_and_update_lr()
                        scheduler.zero_grad()

                        log_loss_num += 1

                        loss_num = 0

                        op_size = min(random.choice(OPTIMIZE_WINDOW_SIZE), doc_length)
                        if doc_length < MAX_OPTIMIZE_WINDOW_SIZE:
                            op_size = doc_length

                        if update_step % LOG_STEP == 0:
                            with open(OUTPUT_FILE, 'a') as fw:
                                fw.write("Update Steps {} loss: {}\n".format(update_step, epoch_loss / num_batches))
                            print("Update Steps {} loss: {}\n".format(update_step, epoch_loss / num_batches))
                        if update_step % EVAL_STEP == 0:
                            do_eval = True

                if loss_num > 0:
                    update_step += 1
                    num_batches += 1
                    log_loss_num += 1
                    scheduler.step_and_update_lr()
                    scheduler.zero_grad()

                    if update_step % LOG_STEP == 0:
                        with open(OUTPUT_FILE, 'a') as fw:
                            fw.write("Update Steps {} loss: {}\n".format(update_step, epoch_loss / num_batches))
                        print("Update Steps {} loss: {}\n".format(update_step, epoch_loss / num_batches))
                    if update_step % EVAL_STEP == 0:
                        do_eval = True

                if do_eval:
                    torch.save(model.state_dict(), "./ckpt-{}-{}-{}".format(TRAINING_MODE, DSNAME, update_step))
                    with torch.no_grad():
                        model.eval()
                        evaluate_with_order(model, test_dataloader, ids_to_vocab, OUTPUT_FILE, False)

                    model.train()
                    model.reset_context_state()
                    model.set_decode(False)

            # if do_eval:
            if update_step >= MIN_STEP:
                torch.save(model.state_dict(), "./ckpt-{}-{}-{}".format(TRAINING_MODE, DSNAME, update_step))
                with torch.no_grad():
                    model.eval()
                    evaluate_with_order(model, test_dataloader, ids_to_vocab, OUTPUT_FILE, False)

                model.train()
                model.reset_context_state()
                model.set_decode(False)


main()
