import os
import cv2
import glob
import random
import numpy as np

BACKGROUND_SAMPLING = 0.25
DATASET_PATH = "{0}/dataset/highway/input".format(os.getcwd())
print DATASET_PATH

if __name__ == '__main__':
    background_samples = glob.glob("{0}/*.jpg".format(DATASET_PATH))
    
    K = int(len(background_samples) * BACKGROUND_SAMPLING)

    background_random_samples = random.sample(background_samples, K)

    background_random_sample = cv2.imread(background_random_samples[0])
    background_mean = np.float32(background_random_sample)
    
    for background_random_sample in background_random_samples[1:]:
        background_sample = cv2.imread(background_random_sample)
        background_mean += np.float32(background_sample)

    background_mean = background_mean / K
    background_mean = cv2.convertScaleAbs(background_mean)

    background_mean_1ch = cv2.cvtColor(background_mean, cv2.COLOR_BGR2GRAY)
    background_mean_3ch = background_mean

    cv2.imshow('background_mean_1ch', background_mean_1ch)
    cv2.imshow('background_mean_3ch', background_mean_3ch)
    
    cv2.imwrite('background_mean_1ch.png', background_mean_1ch)
    cv2.imwrite('background_mean_3ch.png', background_mean_3ch)

    cv2.waitKey(0)
    cv2.destroyAllWindows()