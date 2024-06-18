import numpy as np
import pickle
import os
from tqdm import tqdm

from datasets.seven_scenes import TrainingTuple
from utils.utils import parse


rootDir = "/home/david/datasets/"
iouDir = "/home/david/datasets/iou_"


def gen_tuple(train, scene, seqLength):
    queries = {}
    sceneDir = os.path.join(rootDir, scene)
    rgbDir = os.path.join(sceneDir, "color")
    if train:
        iou = np.load(os.path.join(iouDir, "train_" + scene + "_iou.npy"))
        poseDir = os.path.join(rootDir, scene, "train", "pose")
        filePath = os.path.join("./", scene + "_train.pickle")
    else:
        iou = np.load(os.path.join(iouDir, "test_" + scene + "_iou.npy"))
        poseDir = os.path.join(rootDir, scene, "test", "pose")
        filePath = os.path.join("./", scene + "_test.pickle")

    poseList = os.listdir(poseDir)
    poseList.sort()

    rgbList = [
        os.path.join(rgbDir, p.split("/")[-1].replace("pose.txt", "color.png"))
        for p in poseList
    ]

    for anchorNdx in tqdm(range(iou.shape[0])):
        positives = []
        negatives = []
        nonNegatives = []
        labels = list(range(iou.shape[0]))
        for ndx in range(iou.shape[1]):
            if ndx == anchorNdx or (ndx // seqLength) == (anchorNdx // seqLength):
                nonNegatives.append(ndx)
                continue
            sub = iou[anchorNdx][ndx] - iou[ndx][anchorNdx]
            sum = iou[anchorNdx][ndx] + iou[ndx][anchorNdx]

            if abs(sub) < 0.3:
                if sum > 0.4:
                    positives.append(ndx)
                    nonNegatives.append(ndx)
                elif sum > 0.05:
                    nonNegatives.append(ndx)
            if (iou[anchorNdx][ndx] > 0.01 or iou[ndx][anchorNdx] > 0.01) and (
                len(nonNegatives) == 0 or nonNegatives[-1] != ndx
            ):
                nonNegatives.append(ndx)
        nonNegatives = np.sort(list(set(nonNegatives)))
        positives = np.sort(positives)
        negatives = np.setdiff1d(labels, nonNegatives, True).tolist()
        queries[anchorNdx] = TrainingTuple(
            id=anchorNdx,
            timestamp=anchorNdx,
            rel_scan_filepath=rgbList[anchorNdx],
            pose=np.ndarray([]),
            positives=positives,
            non_negatives=nonNegatives,
        )
    with open(filePath, "wb") as f:
        pickle.dump(queries, f)


def main():
    parser = parse()
    scene = parser.scene
    seqLength = parser.length
    gen_tuple(True, scene, seqLength)
    gen_tuple(False, scene, seqLength)

    return


if __name__ == "__main__":
    main()
