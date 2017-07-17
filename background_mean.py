import os
import cv2
import glob
import random
import numpy as np

BACKGROUND_SAMPLING = 0.25
# DATASET_BASE_PATH = "{0}/dataset".format(os.getcwd())
DATASET_BASE_PATH = "{0}/dataset".format("/Users/lvmc/Desktop")

DATASETS = {
    "PTZ": ["continuousPan", "intermittentPan", "twoPositionPTZCam","zoomInZoomOut"],
    "badWeather": ["blizzard", "skating", "snowFall", "wetSnow"],
    "baseline": ["PETS2006", "highway", "office", "pedestrians"],
    "cameraJitter": ["badminton", "boulevard", "sidewalk", "traffic"],
    "dynamicBackground": ["boats", "canoe", "fall", "fountain01", "fountain02", "overpass"],
    "intermittentObjectMotion": ["abandonedBox", "parking", "sofa", "streetLight", "tramstop", "winterDriveway"],
    "lowFramerate": ["port_0_17fps", "tramCrossroad_1fps", "tunnelExit_0_35fps", "turnpike_0_5fps"],
    "nightVideos": ["bridgeEntry", "busyBoulvard", "fluidHighway", "streetCornerAtNight", "tramStation", "winterStreet"],
    "shadow": ["backdoor", "bungalows", "busStation", "copyMachine", "cubicle", "peopleInShade"],
    "thermal": ["corridor", "diningRoom", "lakeSide", "library", "park"],
    "turbulence": ["turbulence0", "turbulence1", "turbulence2", "turbulence3"],
}

DEBUG = False

def main():
    for key, val in DATASETS.iteritems():
        for v in val:
            dataset_path = "{0}/{1}/{2}/input".format(DATASET_BASE_PATH, key, v)

            background_sample_filenames = glob.glob("{0}/*.jpg".format(dataset_path))
            
            K = int(len(background_sample_filenames) * BACKGROUND_SAMPLING)

            background_samples = random.sample(background_sample_filenames, K)

            background_sample = cv2.imread(background_samples[0])
            background_mean = np.float64(background_sample)
            
            for background_sample in background_samples[1:]:
                background_sample = cv2.imread(background_sample)
                background_mean += np.float64(background_sample)

            background_mean = background_mean / K
            background_mean = cv2.convertScaleAbs(background_mean)

            background_mean_3ch = cv2.convertScaleAbs(background_mean)
            background_mean_1ch = cv2.cvtColor(background_mean, cv2.COLOR_BGR2GRAY)

            cv2.imwrite("background_{0}_{1}_1ch.jpg".format(key, v), background_mean_1ch)
            cv2.imwrite("background_{0}_{1}_3ch.jpg".format(key, v), background_mean_3ch)

            if (DEBUG):
                cv2.imshow('background_1ch', background_mean_1ch)
                cv2.imshow('background_3ch', background_mean_3ch)
                cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
