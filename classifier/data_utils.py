import os
import collections
import pandas as pd
import pickle
import numpy as np
from khaiii import KhaiiiApi

api = KhaiiiApi()

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
TEST_GEN_PATH = "../data/save_generator/text_sentenceGenerate.txt"
TEST_GEN_RESULT_PATH = '../data/save_classifier/classi_Result.txt'

def coverNieun(line):
    newSentence = []
    line = line.split(' ')

    for i in line:
        # ㄴ 합병 이벤트 여부
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

def clean_str(text):
    text = text.strip().split(' ')
    if text[0][0]=='\ufeff':
        text[0] = text[0][1:]

    return text


def build_word_dict():
    if not os.path.exists("word_dict.pickle"):
        train_df = pd.read_csv(TRAIN_PATH, names=["title", "view"])
        titles = train_df["title"]

        max_length = 0
        words = list()
        for title in titles:
            parseline = clean_str(title)
            if max_length < len(parseline):
                max_length = len(parseline)
            for word in parseline:
                words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        id_dict = dict()
        word_dict["<pad>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<eos>"] = 2
        id_dict["0"] = "<pad>"
        id_dict["1"] = "<unk>"
        id_dict["2"] = "<eos>"
        for word, _ in word_counter:
            temp_num = len(word_dict)
            word_dict[word] = temp_num
            id_dict[str(temp_num)] = word

        # return max_length + 1 because of <eos> token.
        max_length += 1

        with open("word_dict.pickle", "wb") as f:
            pickle.dump([word_dict, id_dict, max_length], f)

    else:
        with open("word_dict.pickle", "rb") as f:
            word_dict, id_dict, max_length = pickle.load(f)

    return word_dict, id_dict, max_length


def build_word_dataset(step, word_dict, document_max_len):
    if step == "train":
        df = pd.read_csv(TRAIN_PATH, names=["title", "view"])
    else:
        df = pd.read_csv(TEST_PATH, names=["title", "view"])

    # Shuffle dataframe

    df = df.sample(frac=1)
    x = list(map(lambda d: clean_str(d), list(df["title"])))
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d + [word_dict["<eos>"]], x))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [word_dict["<pad>"]], x))

    hitInt = list()
    for i in list(df["view"]):
        if i == 'non-hit':
            hitInt.append(0)
        elif i == 'hit':
            hitInt.append(1)
        else:
            print('exception :', i)
    y = hitInt
    return x, y


def build_word_dataset_exceptY(word_dict, document_max_len, customSentence=False):

    # Shuffle dataframe

    with open(TEST_GEN_PATH, 'r', encoding='utf-8-sig') as f:
        if not customSentence:
            lines = f.readlines()
        else:
            words = ''
            for word in api.analyze(customSentence):
                part = str(word).split('\t')[1]
                while ' + ' in part:
                    index = part.find(' + ')
                    part = part[:index] + part[index + 2:]
                words += ' ' + part
            lines = [words.strip()]

        x = list(map(lambda d: clean_str(d), list(lines)))
        x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
        x = list(map(lambda d: d + [word_dict["<eos>"]], x))
        x = list(map(lambda d: d[:document_max_len], x))
        x = list(map(lambda d: d + (document_max_len - len(d)) * [word_dict["<pad>"]], x))

        x2 = list()
        for i in x:
            if i not in x2:
                x2.append(i)
        print(len(x2))
        return x2

def build_char_dataset(step, model, document_max_len):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’'\"/|_#$%ˆ&*˜‘+=<>()[]{} "
    if step == "train":
        df = pd.read_csv(TRAIN_PATH, names=["class", "title", "content"])
    else:
        df = pd.read_csv(TEST_PATH, names=["class", "title", "content"])

    # Shuffle dataframe
    df = df.sample(frac=1)

    char_dict = dict()
    char_dict["<pad>"] = 0
    char_dict["<unk>"] = 1
    for c in alphabet:
        char_dict[c] = len(char_dict)

    alphabet_size = len(alphabet) + 2

    x = list(map(lambda content: list(map(lambda d: char_dict.get(d, char_dict["<unk>"]), content.lower())), df["content"]))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [char_dict["<pad>"]], x))

    y = list(map(lambda d: d - 1, list(df["class"])))

    return x, y, alphabet_size


def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]


def batch_iter_exceptY(inputs, batch_size, num_epochs):
    output = []
    num_batches_per_epoch = len(inputs) // batch_size + 1
    output.append('cnt:' + (str)(num_epochs * num_batches_per_epoch))

    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            output.append(inputs[start_index:end_index])
    return output

def id_to_word(idlist, id_dict):
    textAndLogits = list()
    for i in idlist:
        ids = i[0]
        text = ''
        for id in ids:
            word = id_dict[str(id)]
            if '/' in word:
                word = word[:word.rfind('/')]
            if word == "<eos>":
                break
            text += ' ' + word
        textAndLogits.append(text.strip() + '\t' + str(round(float(i[1] * 100),2)))

    with open(TEST_GEN_RESULT_PATH, 'w', encoding='utf-8-sig') as f:
        for i in textAndLogits:
            f.write(str(i) + '\n')

    return textAndLogits[0:10]
