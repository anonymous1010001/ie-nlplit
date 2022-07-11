from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import math
import os
import random
import shutil
import sys
import pickle as pkl

import numpy as np
import tensorflow as tf
from fastprogress import master_bar, progress_bar
from seqeval.metrics import classification_report
from tqdm import tqdm

from model import BertNer
#from optimization import AdamWeightDecay, WarmUp
# from tokenization import FullTokenizer
from transformers import BertTokenizer, TFBertModel
from transformers import AutoTokenizer, TFAutoModel
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased", from_pt=True)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask

def get_fg_label_to_BIO(y_true,y_pred,dict_new):
  y_true_new=[]
  y_pred_new=[]
  for i,j in enumerate(tqdm(y_true)):
    y_true_temp=[]
    y_pred_temp=[]
    for k,l in enumerate(j):
      y_true_temp.append(dict_new[l])
      y_pred_temp.append(dict_new[y_pred[i][k]])
    y_true_new.append(y_true_temp)
    y_pred_new.append(y_pred_temp)
  return y_true_new,y_pred_new


def get_clean_data(data):
  sentence_lst={}
  sentence_lst_={}
  data_new=[]
  data_new_=[]
  for i,j in enumerate(tqdm(data)):
    if ' '.join(j[0]) not in sentence_lst:
      sentence_lst[' '.join(j[0])]=j[-1]
      data_new.append((j[0],j[-1]))
    else:
      sentence_lst_[' '.join(j[0])]=j[-1]
      data_new_.append((j[0],j[-1]))
  if len(data_new)>len(data_new_):
    return data_new
  else:
    return data_new_



def readfile(filename):
    f = open(filename)
    data = []
    sentence = []
    label = []
    count=0
    for i,line in enumerate(f):
      splits = line.split('\t')
      if len(splits)==1:
        continue
      if len(line) == 0 or line.startswith('Sentence:') or line[0] == "\n":
        if len(sentence) > 0:
          data.append((sentence, label))
          sentence = []
          label = []
          splits = line.split('\t')
          #print(splits)
          sentence.append(splits[1])
          label.append(splits[-1].strip())
          continue
        elif len(sentence)==0:
          splits = line.split('\t')
          sentence.append(splits[1])
          label.append(splits[-1].strip())
      else:
        splits = line.split('\t')
        #print(splits)
        sentence.append(splits[1])
        label.append(splits[-1].strip())
    #print(len(sentence))
    if len(sentence) > 0:
      data.append((sentence, label))
      sentence = []
      label = []
    data_cln=get_clean_data(data)
    data_cln_srt=sorted(data_cln)
    return data_cln_srt

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["O", "B", "I", "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        start_position = 1
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            if len(token)>=1:
              tokens.extend(token)
              label_1 = labellist[i]
              labels.append(label_1)
              valid.append(start_position)
              start_position += len(token)
              label_mask.append(True)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 0)
        label_mask.insert(0, True)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(valid[-1]+1)
        label_mask.append(True)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [True] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            label_mask.append(False)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(False)
        while len(valid) < max_seq_length:
            valid.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask))
    return features


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--label_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input label dir. Should contain the .pkl files for the task.")
    parser.add_argument("--label_to_BIO",
                        default=None,
                        type=str,
                        help="The input label to BIO tag dir. Should contain the .tsv files (or other data files) for the task.")
    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--do_BIOconv",
                        default=False,
                        help="Whether to run mapping algo or not.")

    args = parser.parse_args()

    processor = NerProcessor()
    label_list = pkl.load(open(args.label_dir,"rb"))
    num_labels = len(label_list) + 1

    label_map = {i: label for i, label in enumerate(label_list, 1)}

    if args.do_eval:
        # load tokenizer
        #tokenizer = FullTokenizer(os.path.join(args.output_dir, "vocab.txt"), args.do_lower_case)
        # model build hack : fix
        config = json.load(open(os.path.join(args.output_dir,"bert_config.json")))
        ner = BertNer(config, tf.float32, num_labels, args.max_seq_length)
        ids = tf.ones((1,128),dtype=tf.int32)
        _ = ner(ids,ids,ids,ids, training=False)
        ner.load_weights(os.path.join(args.output_dir,"model.h5"))
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evalution *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.input_ids for f in eval_features],dtype=np.int32))
        all_input_mask = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.input_mask for f in eval_features],dtype=np.int32))
        all_segment_ids = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.segment_ids for f in eval_features],dtype=np.int32))
        all_valid_ids = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.valid_ids for f in eval_features],dtype=np.int32))

        all_label_ids = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.label_id for f in eval_features],dtype=np.int32))

        eval_data = tf.data.Dataset.zip(
            (all_input_ids, all_input_mask, all_segment_ids, all_valid_ids, all_label_ids))
        batched_eval_data = eval_data.batch(args.eval_batch_size)

        y_true=[]
        y_pred=[]
        y_true_ids=[]
        y_pred_ids=[]
        count=0
        for k,(input_ids, input_mask, segment_ids, valid_ids, label_ids) in enumerate(tqdm(batched_eval_data)):
            logits = ner(input_ids, input_mask,segment_ids, valid_ids, training=False)
            logits = tf.argmax(logits,axis=2)
            #print(logits.shape)
            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                temp_1_ids = []
                temp_2_ids = []
                for j,m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_ids[i][j].numpy() == len(label_map):
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        y_true_ids.append(temp_1_ids)
                        y_pred_ids.append(temp_2_ids)
                        count=count+1
                        break
                    else:
                        temp_1.append(label_map[label_ids[i][j].numpy()])
                        temp_2.append(label_map[logits[i][j].numpy()])
                        temp_1_ids.append(label_ids[i][j].numpy())
                        temp_2_ids.append(logits[i][j].numpy())
        report = classification_report(y_true, y_pred,digits=4)
        output_eval_file = os.path.join(args.output_dir, "test_results_all.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test results *****")
            logger.info("\n%s", report)
            writer.write(report)

        if args.do_BIOconv:
            dict_new=open(args.label_to_BIO,"rb")
            y_true_new,y_pred_new=get_fg_label_to_BIO(y_true,y_pred,dict_new)
            report_dummy = classification_report(y_true_new, y_pred_new,digits=4)
            output_eval_file_1 = os.path.join(args.output_dir, "test_results_actual.txt")
            with open(output_eval_file_1, "w") as writer_1:
                logger.info("***** Test results *****")
                logger.info("\n%s", report_dummy)
                writer_1.write(report_dummy)
if __name__ == "__main__":
    main()