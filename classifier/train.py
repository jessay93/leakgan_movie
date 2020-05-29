import argparse
import tensorflow as tf
from data_utils import *
from sklearn.model_selection import train_test_split
from cnn_models.word_cnn import WordCNN
from cnn_models.char_cnn import CharCNN
from cnn_models.vd_cnn import VDCNN
from rnn_models.word_rnn import WordRNN
from rnn_models.attention_rnn import AttentionRNN
from rnn_models.rcnn import RCNN

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.8
config.gpu_options.allow_growth=True

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="word_cnn",
                    help="word_cnn | char_cnn | vd_cnn | word_rnn | att_rnn | rcnn")
args = parser.parse_args()

NUM_CLASS = 2
BATCH_SIZE = 256
NUM_EPOCHS = 20
WORD_MAX_LEN = 50
CHAR_MAX_LEN = 1014
args.model = "att_rnn"

print("Building dataset...")
if args.model == "char_cnn":
    x, y, alphabet_size = build_char_dataset("train", "char_cnn", CHAR_MAX_LEN)
elif args.model == "vd_cnn":
    x, y, alphabet_size = build_char_dataset("train", "vdcnn", CHAR_MAX_LEN)
else:
    word_dict, _, WORD_MAX_LEN = build_word_dict()
    vocabulary_size = len(word_dict)
    x, y = build_word_dataset("train", word_dict, WORD_MAX_LEN)

    print('WORD_MAX_LEN :', WORD_MAX_LEN)

train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=1/9)


with tf.Session(config=config) as sess:
    if args.model == "word_cnn":
        model = WordCNN(vocabulary_size, WORD_MAX_LEN, NUM_CLASS)
    elif args.model == "char_cnn":
        model = CharCNN(alphabet_size, CHAR_MAX_LEN, NUM_CLASS)
    elif args.model == "vd_cnn":
        model = VDCNN(alphabet_size, CHAR_MAX_LEN, NUM_CLASS)
    elif args.model == "word_rnn":
        model = WordRNN(vocabulary_size, WORD_MAX_LEN, NUM_CLASS)
    elif args.model == "att_rnn":
        model = AttentionRNN(vocabulary_size, WORD_MAX_LEN, NUM_CLASS)
    elif args.model == "rcnn":
        model = RCNN(vocabulary_size, WORD_MAX_LEN, NUM_CLASS)
    else:
        raise NotImplementedError()

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    train_batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS)
    num_batches_per_epoch = (len(train_x) - 1) // BATCH_SIZE + 1
    max_accuracy = 0

    for x_batch, y_batch in train_batches:
        train_feed_dict = {
            model.x: x_batch,
            model.y: y_batch,
            model.is_training: True
        }

        _, step, loss = sess.run([model.optimizer, model.global_step, model.loss], feed_dict=train_feed_dict)

        if step % 100 == 0:
            print("step {0}: loss = {1}".format(step, loss))

        if step % 100 == 0:
            # Test accuracy with validation data for each epoch.
            valid_batches = batch_iter(valid_x, valid_y, BATCH_SIZE, 1)
            sum_accuracy, cnt = 0, 0

            sampletrigger = True
            for valid_x_batch, valid_y_batch in valid_batches:
                valid_feed_dict = {
                    model.x: valid_x_batch,
                    model.y: valid_y_batch,
                    model.is_training: False
                }
                if sampletrigger:
                    sampletrigger = False
                    accuracy, y_sample, predictions_sample = sess.run([model.accuracy, model.y, model.predictions], feed_dict=valid_feed_dict)
                    for i in range(min(len(y_sample), 10)):
                        print('실제값 :', 'non-hit' if y_sample[i] == 0 else 'hit', ', 예측값 :', 'non-hit' if predictions_sample[i] == 0 else 'hit')
                else:
                    accuracy = sess.run(model.accuracy, feed_dict=valid_feed_dict)
                sum_accuracy += accuracy
                cnt += 1
            valid_accuracy = sum_accuracy / cnt

            print("\nValidation Accuracy = {1}\n".format(step // num_batches_per_epoch, sum_accuracy / cnt))

            # Save model
            if valid_accuracy > max_accuracy:
                max_accuracy = valid_accuracy
                saver.save(sess, "{0}/{1}.ckpt".format(args.model, args.model), global_step=step)
                print("Model is saved.\n")
