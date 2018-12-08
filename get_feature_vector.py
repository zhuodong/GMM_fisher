import sys, glob, argparse
import numpy as np
import math, cv2
from scipy.stats import multivariate_normal
import time
from sklearn import svm
from cv2 import ml
from sklearn.externals import joblib

fisher_data = open("fisher_data1.txt", 'w')

def likelihood_moment(x, ytk, moment):
    x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
    return x_moment * ytk


def likelihood_statistics(samples, means, covs, weights):
    gaussians, s0, s1, s2 = {}, {}, {}, {}
    samples = zip(range(0, len(samples)), samples)
    g = [multivariate_normal(mean=means[k], cov=covs[k]) for k in range(0, len(weights))]
    for k in range(0, len(weights)):
        s0[k], s1[k], s2[k] = 0, 0, 0
        for index, x in samples:
            gaussians[index] = np.array([g_k.pdf(x) for g_k in g])
            probabilities = np.multiply(gaussians[index], weights)
            probabilities = probabilities / np.sum(probabilities)
            s0[k] = s0[k] + likelihood_moment(x, probabilities[k], 0)
            s1[k] = s1[k] + likelihood_moment(x, probabilities[k], 1)
            s2[k] = s2[k] + likelihood_moment(x, probabilities[k], 2)
    return s0, s1, s2


def fisher_vector_weights(s0, s1, s2, means, covs, w, T):
    return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k])) for k in range(0, len(w))])


def fisher_vector_means(s0, s1, s2, means, sigma, w, T):
    return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])


def fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
    return np.float32(
        [(s2[k] - 2 * means[k] * s1[k] + (means[k] * means[k] - sigma[k]) * s0[k]) / (np.sqrt(2 * w[k]) * sigma[k]) for
         k in range(0, len(w))])


def normalize(fisher_vector):
    v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
    return v / np.sqrt(np.dot(v, v))


def fisher_vector(samples, means, covs, w):
    # samples为数据的sift特征向量，means, covs, w高斯混合模型参数
    s0, s1, s2 = likelihood_statistics(samples, means, covs, w)
    T = samples.shape[0]
    covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
    a = fisher_vector_weights(s0, s1, s2, means, covs, w, T)
    b = fisher_vector_means(s0, s1, s2, means, covs, w, T)
    c = fisher_vector_sigma(s0, s1, s2, means, covs, w, T)
    # print("a",a)
    # print("np.concatenate(b)", np.concatenate(b))
    # print("np.concatenate(c)", np.concatenate(c))
    fv = np.concatenate([a, np.concatenate(b), np.concatenate(c)])
    fv = normalize(fv)
    return fv


# 从文件夹中获取
def get_fisher_vectors_from_folder(folder):
    files = glob.glob(folder + "/*.bmp")
    for file in files:
        print("files", file)
        label = file[::-1].split('/', 2)[0][::-1][0]
        gmm = load_gmm('./GMMpara')
        data = fisher_vector(image_descriptors(file), *gmm)
        # print("fisher_vector",fisher_data)
        for ip in data:
            fisher_data.write(str(ip))
            fisher_data.write(',')
        fisher_data.write(label)
        fisher_data.write('\n')
    return np.float32(data)


# 获取fisher特征
def fisher_features(folder):
    folders = glob.glob(folder + "/*")
    features = {f: get_fisher_vectors_from_folder("/".join(f.split("\\"))) for f in folders}
    print("features", features)
    return features


def load_gmm(folder=""):
    files = ["means.gmm.npy", "covs.gmm.npy", "weights.gmm.npy"]

    return map(lambda file: np.load(file), map(lambda s: folder + "/" + s, files))


# 提取数据的sift特征
def image_descriptors(file):
    img = cv2.imread(file, 0)
    img = cv2.resize(img, (256, 256))
    sift = cv2.xfeatures2d.SIFT_create()
    _, kp = sift.detectAndCompute(img, None)
    # print("kp",kp)
    return kp

if __name__ == '__main__':
    # 数据集文件夹
    working_folder = "train_valid_test(700_300)/all_train_valid"
    # 加载参数
    gmm = load_gmm('./GMMpara')
    print("参数加载完成！")
    # 获取fisher特征
    fisher_features = fisher_features(working_folder)
    print("fisher_features", fisher_features)



