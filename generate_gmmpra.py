import sys, glob, argparse
from cv2 import ml
import numpy as np
import cv2

#提取数据的sift特征
def image_descriptors(file):
	img = cv2.imread(file, 0)
	img = cv2.resize(img, (256, 256))
	#_ , descriptors = cv2.SIFT().detectAndCompute(img, None)
	sift = cv2.xfeatures2d.SIFT_create()
	_,kp = sift.detectAndCompute(img,None)
	#print("kp",kp)
	print("sift的长度：",len(kp))
	return kp

def dictionary(descriptors, N):
	#em = cv2.EM(N)
	#em.train(descriptors)
	em = ml.EM_create()
	em.setClustersNumber(N)
	print("EM开始训练！")
	em.trainEM(descriptors)
	print("EM训练完成！")
	means = em.getMeans()
	covs = em.getCovs()
	weights = em.getWeights()
	return np.float32(means), np.float32(covs), np.float32(weights)[0]

#对文件夹下的每一个数据进行处理	
def folder_descriptors(folder):
	print("folder",folder)
	files = glob.glob(folder + "/*.bmp")	
	print("Calculating descriptos. Number of images is", len(files))
	print( )
	
	return np.concatenate([image_descriptors(file) for file in files])


def generate_gmm(input_folder, N):
	print("input_folder",input_folder)
	words = np.concatenate([folder_descriptors("/".join(folder.split("\\"))) for folder in glob.glob(input_folder + '/*')])
	print("words",len(words))
	print("Training GMM of size", N)
	means, covs, weights = dictionary(words, N)
	#Throw away gaussians with weights that are too small:
	th = 1.0 / N
	means = np.float32([m for k,m in zip(range(0, len(weights)), means) if weights[k] > th])
	covs = np.float32([m for k,m in zip(range(0, len(weights)), covs) if weights[k] > th])
	weights = np.float32([m for k,m in zip(range(0, len(weights)), weights) if weights[k] > th])
	print('weights',len(weights))
	np.save("GMMpara/means.gmm", means)
	np.save("GMMpara/covs.gmm", covs)
	np.save("GMMpara/weights.gmm", weights)
	print("参数保存成功！")
	return means, covs, weights
	
if __name__ == '__main__':

	#数据集文件夹
	working_folder = "train_valid_test(700_300)/all_train_valid"
	#num为单词字典的数目
	number = 5
	#使用EM对sift获取的特征进行训练，生成EM参数
	generate_gmm(working_folder, number)
	print("特征参数训练完成！")
	
	
	
	
	
