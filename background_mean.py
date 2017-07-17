import os
import cv2
import glob
import random
import numpy as np

BACKGROUND_SAMPLING = 0.25
DATASET_PATH = "{0}/dataset/highway/input".format(os.getcwd())

def main():
    background_sample_filenames = glob.glob("{0}/*.jpg".format(DATASET_PATH))
    
    K = int(len(background_sample_filenames) * BACKGROUND_SAMPLING)

    background_samples = random.sample(background_sample_filenames, K)

    background_sample = cv2.imread(background_samples[0])
    background_mean = np.float32(background_sample)
    
    for background_sample in background_samples[1:]:
        background_sample = cv2.imread(background_sample)
        background_mean += np.float32(background_sample)

    background_mean = background_mean / K
    background_mean = cv2.convertScaleAbs(background_mean)

    background_mean_3ch = cv2.convertScaleAbs(background_mean)
    background_mean_1ch = cv2.cvtColor(background_mean, cv2.COLOR_BGR2GRAY)

    cv2.imshow('background_1ch', background_mean_1ch)
    cv2.imshow('background_3ch', background_mean_3ch)
    
    cv2.imwrite('background_1ch.png', background_mean_1ch)
    cv2.imwrite('background_3ch.png', background_mean_3ch)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
