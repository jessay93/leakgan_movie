import tensorflow as tf
import argparse
from data_utils import *
import numpy as np

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.8
config.gpu_options.allow_growth=True

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="word_cnn",
                    help="word_cnn | char_cnn | vd_cnn | word_rnn | att_rnn | rcnn")
args = parser.parse_args()

args.model = "att_rnn"

BATCH_SIZE = 64
WORD_MAX_LEN = 100
CHAR_MAX_LEN = 1014

# task = 'claData'   : 저장된 텍스트 파일을 읽고, 각 문장 별 hit 점수를 예측
# task = 'hitData'   : 한 문장(inputText)에 대한 hit 점수를 확인
task = 'claData'
inputText = "오늘 기생충 보고 왔습니다(스포 X)"

def softmax_right(x0, x1):
    e_x0 = np.exp(x0)
    e_x1 = np.exp(x1)
    return float(e_x1 / sum([e_x0, e_x1]))

word_dict, id_dict, WORD_MAX_LEN = build_word_dict()

checkpoint_file = tf.train.latest_checkpoint(args.model)
graph = tf.Graph()
with graph.as_default():
    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        print("start sentence classification!!")

        if task == 'claData':
            test_x = build_word_dataset_exceptY(word_dict, WORD_MAX_LEN)
        elif task == 'hitData':
            test_x = build_word_dataset_exceptY(word_dict, WORD_MAX_LEN, inputText)

        x = graph.get_operation_by_name("x").outputs[0]
        is_training = graph.get_operation_by_name("is_training").outputs[0]
        logits = graph.get_operation_by_name("output/logits/BiasAdd").outputs[0]
        batches = batch_iter_exceptY(test_x, BATCH_SIZE, 1)

        guessList = list()
        cnt = 0
        batch_length = 0

        for batch_x in batches:
            if len(batch_x) > 4 and batch_x[0:4] == 'cnt:':
                batch_length = (int)(batch_x[4:])
            else:
                cnt += 1
                feed_dict = {
                    x: batch_x,
                    is_training: False
                }
                logits_out = sess.run(logits, feed_dict=feed_dict)
                for i in range(len(logits_out)):
                    guessList.append([batch_x[i], softmax_right(logits_out[i][0], logits_out[i][1])])

        guessList = sorted(guessList, key=lambda hit: hit[1], reverse=True)
        message = id_to_word(guessList, id_dict)
        print('message :', message)
        print("end sentence generate!!")
