import numpy as np
import sys


def read_emb(emb_path):
    file = open(emb_path, mode='r', encoding='utf-8', errors="replace")
    dim = int(file.readline().split(' ')[1])
    word_vec_dict = {}
    line = str(file.readline()).strip()
    while len(line) > 1:
        word, vec = line.split(' ', 1)
        if word.lower() not in word_vec_dict:
            word_vec_dict[word.lower()] = np.fromstring(vec, sep=' ', dtype=float)
        else:
            None # print("already", word)
        line = str(file.readline()).strip()
    file.close()
    return word_vec_dict, dim


def read_m_analogy_set(file_path):
    with open(file_path, 'r') as file_in:
        line = str(file_in.readline()).strip()
        words = []
        while len(line) > 1:
            left, right = line.split()
            words = words + [left.lower(), right.lower()]
            line = str(file_in.readline()).strip()
    return words


def filter_x_embs(X_word_vec_dict, Y_word_vec_dict, X_analogy_words, Y_analogy_words, dim):
    assert len(X_analogy_words) == len(Y_analogy_words)

    X_mat = np.empty((0, dim), dtype=float)
    Y_mat = np.empty((0, dim), dtype=float)

    for i in range(len(X_analogy_words)):
        if (X_analogy_words[i] in X_word_vec_dict) and (Y_analogy_words[i] in Y_word_vec_dict):
            X_mat = np.vstack([X_mat, X_word_vec_dict[X_analogy_words[i]]])
            Y_mat = np.vstack([Y_mat, Y_word_vec_dict[Y_analogy_words[i]]])
    return X_mat, Y_mat


def preprocess_emb(emb_mat):
    mu = emb_mat.mean(0)

    # mean centring
    emb_mat0 = emb_mat - mu

    # row-wise L2 norm
    norm = np.sqrt((emb_mat0 ** 2.).sum()) / len(emb_mat)

    # scale to equal (unit) norm
    return emb_mat0 / norm


def main(input_list):
    [A_emb_path, B_emb_path, A_analogy_path, B_analogy_path, A_mat_path, B_mat_path] = input_list
    A_analogy_words = read_m_analogy_set(A_analogy_path)
    B_analogy_words = read_m_analogy_set(B_analogy_path)

    A_word_vec_dict, A_dim = read_emb(A_emb_path)
    B_word_vec_dict, B_dim = read_emb(B_emb_path)

    A_mat, B_mat = filter_x_embs(A_word_vec_dict, B_word_vec_dict, A_analogy_words, B_analogy_words, A_dim)

    np.save(A_mat_path, preprocess_emb(A_mat))
    np.save(B_mat_path, preprocess_emb(B_mat))


main(sys.argv[1:])
