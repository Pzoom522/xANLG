import random
import numpy as np
from scipy import optimize
import argparse


def transport(start_mat, end_mat, metric):
    def dist(x, metric_arg):
        x = np.reshape(x, (1, -1))
        loss = 42
        if metric_arg == 'l1':
            loss = np.sqrt(np.abs(offset_mat - x).sum()) / len(offset_mat)
        elif metric_arg == 'l2':
            loss = np.sqrt(((offset_mat - x) ** 2).sum()) / len(offset_mat)
        elif metric_arg == 'cos':
            dot_sum = np.sum(x * offset_mat, axis=1)
            mod_x = np.sqrt(np.sum(x ** 2))
            mod_mat = np.sqrt(np.sum(offset_mat ** 2))
            loss = np.mean(1 - dot_sum / (mod_x * mod_mat))
        else:
            print('illegal metric!')
        # print(loss)
        return loss

    offset_mat = start_mat - end_mat
    minimum = optimize.fmin(dist, np.mean(offset_mat, axis=0), args=(metric,), xtol=0.0001, ftol=0.0001, disp=False)
    return dist(minimum, metric)


parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, help="monolingual embedding matrix .npy path")
parser.add_argument("-m", type=str, help="metric: {l1, l2, cos}", default="l2")
parser.add_argument("-s", type=int, help="sample number", default=100000)
args = parser.parse_args()

mat = np.load(args.i)

half_len = int(len(mat)/2)
gold = (np.array(range(half_len)) * 2, np.array(range(half_len)) * 2 + 1)
gold_dist = transport(mat[gold[0]], mat[gold[1]], args.m)

print("gold", gold_dist)

indices = list(range(len(mat)))
for cnt in range(args.s):
    random.shuffle(indices)
    sample_dist = transport(mat[tuple([indices[half_len:]])], mat[tuple([indices[:half_len]])], args.m)
    if sample_dist < gold_dist:
        print("better!", indices)
        break

    if cnt % 100 == 0:
        print("finished:", cnt)
