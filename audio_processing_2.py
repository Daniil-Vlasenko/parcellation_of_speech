import numpy as np
import glob
import syllabify
import random as rn
import timit_utils as tu

import audio_processing as ap


# Читаем данные.
corpus = tu.Corpus('../data/TIMIT')
train = corpus.train
test = corpus.test
sentences_train = train.sentences_by_phone_df('aa')
sentences_test = test.sentences_by_phone_df('aa')

# Берем случайно n_test и n_train строк.
n_train = 100
n_test = 100
np.random.seed(0)
indexes_train = np.random.choice(range(sentences_train.shape[0]), n_train, replace=False)
indexes_test = np.random.choice(range(sentences_test.shape[0]), n_test, replace=False)

sentences_train = train.sentences_by_phone_df('aa').sentence[indexes_train]
sentences_test = test.sentences_by_phone_df('aa').sentence[indexes_test]

# Путь к папке, куда сохранять результат обработки.
sentences_path_train = "../data/I_INP/train"
sentences_path_test = "../data/I_INP/test"
I_INP_path = "../data/I_INP/train/I_INP.npy"
Y_path = "../data/I_INP/train/Y.npy"


# Достаем из аудио сырые данные, обрабатываем аудио, уменьшаем частоту, сохраняем отдельно файлы. Позже их объединим.
def processing(sentences, path):
    for count, sent in enumerate(sentences):
        sig = sent.raw_audio
        sig_brain = ap.audioprocessing(sig, plot=False)
        I_INP = np.array([.25 * np.sum(sig_brain[i: i + 4, :], axis=0) for i in range(0, sig_brain.shape[0], 4)]).T
        I_INP = I_INP[::160]
        np.save(f"{path}/I_INP_{count}.npy", I_INP)
        print("I_INP:", count)


# Объединение полученных I_INP для обучения свертки со случайными вставками тишины, сохранение размеров вставок в
# отдельный файл
def concatenate_I_INP(path):
    rn.seed(0)
    intervals = [rn.randint(50, 100)]
    long_sig_brain = np.zeros((intervals[-1], 32))
    sig_brain_files = sorted(glob.glob(path + "/I_INP_*"))
    for count, file in enumerate(sig_brain_files):
        sig_brain = np.load(file)
        intervals.append(rn.randint(50, 100))
        long_sig_brain = np.concatenate((long_sig_brain, sig_brain, np.zeros((intervals[-1], 32))))
        print("I_INP:", count)
    np.save(path + "/intervals.npy", intervals)
    np.save(path + "/I_INP.npy", long_sig_brain)


# Изменение TIMIT на ARPABET
def TIMIT_to_ARPABET(TIMIT_word):
    TIMIT_to_ARPABET = {"ax": "aa", "ax-h": "aa", "axr": "aa", "bcl": "b", "dcl": "d", "dx": "r", "el": "l",
                        "em": "m", "en": "n", "eng": "n", "epi": "-", "gcl": "g", "hv": "hh", "ix": "iy",
                        "kcl": "k", "ng": "n", "nx": "n", "pau": "-", "pcl": "p", "q": "r", "tcl": "t", "ux": "uw"}
    ARPABET_word_ = [TIMIT_to_ARPABET[i].upper() if i in TIMIT_to_ARPABET else i.upper() for i in TIMIT_word]
    return ARPABET_word_


# Перевод предложения в слога
def sentence_to_syllables(sentence):
    phon_df = sentence.phones_df
    sentence_in_phonemes = phon_df.index.tolist()
    sentence_in_phonemes = TIMIT_to_ARPABET(sentence_in_phonemes)
    sentence_in_syllables = syllabify.syllabify(sentence_in_phonemes)
    return sentence_in_syllables


# Вычисление начала слогов на основе таймкодов фонем
def syllables_starts(sentence):
    syllables_starts_ = []
    sentence_in_syllables = sentence_to_syllables(sentence)
    phones_df = sentence.phones_df
    phones_count = -1
    for syllable in sentence_in_syllables:
        is_start = True
        for block in syllable:
            if block == []:
                continue
            else:
                for phon in block:
                    phones_count += 1
                    if is_start and phon != '-':
                        is_start = False
                        syllables_starts_.append(round(phones_df['start'][phones_count] / 160))
    return syllables_starts_


# Вычисление бинарного вектора Y(t) для предложения
def binary_syllables_starts_1(sentence):
    syllables_starts_ = syllables_starts(sentence)
    Y = np.zeros(round(sentence.raw_audio.shape[0] / 160), dtype=int)
    for start in syllables_starts_:
        Y[start + 2] = 1
    return Y


# Вычисление бинарного вектора Y(t) для нескольких предложений с учетом файла вставок тишины
def binary_syllables_starts_2(sentences, path):
    intervals = np.load(path + "/intervals.npy")
    Y = np.zeros(intervals[0], dtype=int)
    for count, sentence in enumerate(sentences):
        Y_tmp = binary_syllables_starts_1(sentence)
        # Корректируем размер Y_tmp с I_INP
        Y_tmp_size = Y_tmp.shape[0]
        I_INP_size = np.load(path + f"/I_INP_{count}.npy").shape[0]
        if I_INP_size > Y_tmp_size:
            Y_tmp = np.concatenate((Y_tmp, np.zeros(I_INP_size - Y_tmp_size, dtype=int)))
        if I_INP_size < Y_tmp_size:
            Y_tmp = Y_tmp[0:-1]
        Y = np.concatenate((Y, Y_tmp, np.zeros(intervals[count + 1], dtype=int)))
    np.save(path + "/Y.npy", Y)


