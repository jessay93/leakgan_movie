#-*- coding: utf-8-sig -*-

import numpy as np
import tensorflow as tf
import random
# from dataloader import Gen_Data_loader, Dis_dataloader
from Discriminator import Discriminator
from LeakGANModel import  LeakGAN
import cPickle
from datetime import datetime
from convertor import convertor
import sys
reload(sys)
sys.setdefaultencoding('utf-8-sig')

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('restore', False, 'Training or testing a model')
flags.DEFINE_boolean('resD', False, 'Training or testing a D model')
flags.DEFINE_string('model', "", 'Model NAME')
#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32  # embedding dimension
HIDDEN_DIM = 32  # hidden state dimension of lstm cell
SEQ_LENGTH = 32  # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 80  # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 48
GOAL_SIZE = 16
STEP_SIZE = 4
#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64

dis_filter_sizes = [2,3]
dis_num_filters = [100,200]
GOAL_OUT_SIZE = sum(dis_num_filters)

dis_dropout_keep_prob = 1.0
dis_l2_reg_lambda = 0.2

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 200
generate_file = '../data/save_generator/result_sentenceGenerate.txt'
pickle_loc = '../data/data/vocab_py2.pkl'
generated_num = 10000
model_path = './ckpts'
maxModelSave = 30


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file, train = 1):
    # Generate Samples
    generated_samples = []
    round = int(generated_num / batch_size) + 1

    for i in range(round):
        generated_samples.extend(trainable_model.generate(sess,1.0,train))

    with open(output_file, 'w') as fout:
        count = 0
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)
            count += 1
            if count == generated_num:
                break

def coverNieun(line):
    newSentence = []
    line = line.split(' ')

    for i in line:

        # ㄴ 합병 이벤트 발생 여부
        merge = False

        if i == "ㄴ":

            # 앞 단어가 있고, 그 단어의 마지막 글자가 한글이면,
            if len(newSentence) != 0 and ord(newSentence[-1][-1]) >= 44032 and ord(newSentence[-1][-1]) < 55200:

                # 앞 단어 마지막 글자에 받침이 없으면,
                if (ord(newSentence[-1][-1]) - 44032) % 28 == 0:
                    # print(chr(ord(newSentence[-1][-1])+4))
                    newSentence[-1] = newSentence[-1][:-1] + chr(ord(newSentence[-1][-1]) + 4)
                    merge = True

                elif (ord(newSentence[-1][-1]) - 44032) % 28 == 17:
                    newSentence[-1] = newSentence[-1][:-1] + chr(ord(newSentence[-1][-1]) - 17) + '운'
                    merge = True

        if not merge:
            newSentence.append(i)

    return ' '.join(newSentence)

def main():
    print 'start time : '
    print datetime.now()
    random.seed(SEED)
    np.random.seed(SEED)

    _, _, _, SEQ_LENGTH, vocab_size = cPickle.load(open(pickle_loc))
    assert START_TOKEN == 0

    discriminator = Discriminator(SEQ_LENGTH,num_classes=2,vocab_size=vocab_size,dis_emb_dim=dis_embedding_dim,filter_sizes=dis_filter_sizes,num_filters=dis_num_filters,
                        batch_size=BATCH_SIZE,hidden_dim=HIDDEN_DIM,start_token=START_TOKEN,goal_out_size=GOAL_OUT_SIZE,step_size=4)
    leakgan = LeakGAN(SEQ_LENGTH,num_classes=2,vocab_size=vocab_size,emb_dim=EMB_DIM,dis_emb_dim=dis_embedding_dim,filter_sizes=dis_filter_sizes,num_filters=dis_num_filters,
                        batch_size=BATCH_SIZE,hidden_dim=HIDDEN_DIM,start_token=START_TOKEN,goal_out_size=GOAL_OUT_SIZE,goal_size=GOAL_SIZE,step_size=4,D_model=discriminator)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver_variables = tf.global_variables()
    saver = tf.train.Saver(saver_variables, max_to_keep=maxModelSave)

    if FLAGS.restore and FLAGS.model:
        if model_path + '/' + FLAGS.model:
            print model_path + '/' + FLAGS.model
            saver.restore(sess, model_path + '/' + FLAGS.model)
        else:
            print "please input all arguments!"
            exit()
    else:
        print "please input all arguments!"
        exit()

    print "start sentence generate!!"
    generate_samples(sess, leakgan, BATCH_SIZE, generated_num, generate_file, 0)
    convertor(generate_file, filedir='../data/save_generator/')
    print "sentenceGenerate.py finish!"

if __name__ == '__main__':
    main()
