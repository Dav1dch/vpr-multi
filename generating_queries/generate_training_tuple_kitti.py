import numpy as np

# import quaternion
import pandas as pd
import pickle
import os
from scipy.spatial import distance
from sklearn.neighbors import KDTree
from datasets.seven_scenes import TrainingTuple
import matplotlib.pyplot as plt

root_dir = "/home/david/datasets/kitti/"

seqs = ["03", "04", "05", "06", "07", "08", "09", "10"]

test_seqs = ["00", "02"]

seq_num = []
test_seq_num = []


for s in seqs:
    # num = len(os.listdir(os.path.join(root_dir, "sequences", s, "velodyne")))
    num = len(np.loadtxt(os.path.join(root_dir, "poses", s + ".txt")))
    seq_num.append(num)
train_cum_sum = np.cumsum(seq_num).tolist()

for s in test_seqs:
    # num = len(os.listdir(os.path.join(root_dir, "sequences", s, "velodyne")))
    num = len(np.loadtxt(os.path.join(root_dir, "poses", s + ".txt")))
    test_seq_num.append(num)
# train_embeddings = np.load('./training/train_apt_embeddings.npy')[0]
# test_embeddings = np.load('./training/test_apt_embeddings.npy')[0]
iou_heaps = []
test_iou_heaps = []
test_cum_sum = np.cumsum(test_seq_num).tolist()

for index, s in enumerate(seqs):
    iou_file = np.load(
        "/home/david/datasets/kitti/" + s + "-full.npz", allow_pickle=True
    )["arr_0"]
    iou_num = seq_num[index]
    iou_heap = [[0 for _ in range(iou_num)] for _ in range(iou_num)]
    for i in iou_file:
        iou_heap[int(i[0])][int(i[1])] = i[2]
    iou_heaps.append(np.array(iou_heap) + np.array(iou_heap).T)
# np.save('train_kitti_iou.npy', iou_heaps)
print(len(iou_heaps))

for index, s in enumerate(test_seqs):
    iou_file = np.load(
        "/home/david/datasets/kitti/" + s + "-full.npz", allow_pickle=True
    )["overlaps"]
    iou_num = test_seq_num[index]
    iou_heap = [[0 for _ in range(iou_num)] for _ in range(iou_num)]
    for i in iou_file:
        iou_heap[int(i[0])][int(i[1])] = i[2]
    test_iou_heaps.append(np.array(iou_heap) + np.array(iou_heap).T)
# np.save('test_kitti_iou.npy', test_iou_heaps)


rgb_list = os.listdir(os.path.join(root_dir, "color"))
rgb_list.sort()
train_rgb_list = rgb_list[sum(test_seq_num) :]

test_rgb_list = rgb_list[: test_cum_sum[0]]

pose_list = os.listdir(os.path.join(root_dir, "pose"))
pose_list.sort()

train_pose_list = pose_list[sum(test_seq_num) :]
test_pose_list = pose_list[: test_cum_sum[0]]

# train_pcl_list = pcl_list[sum(test_seq_num) :]
# # test_pcl_list = pcl_list[test_cum_sum[0]:sum(test_seq_num)]
# test_pcl_list = pcl_list[: test_cum_sum[0]]
# print(len(train_pcl_list))


def gen_tuple(scene):

    queries = {}
    count = 0
    count_non = 0
    if scene == "train":
        gen_rgb_list = train_rgb_list
        gen_pose_list = train_pose_list
        cum_sum = train_cum_sum
    else:
        gen_rgb_list = test_rgb_list
        gen_pose_list = test_pose_list
        # gen_pcl_list = test_pcl_list
        cum_sum = test_cum_sum
    print(len(gen_rgb_list))
    print(len(gen_pose_list))
    labels = list(range(len(gen_rgb_list)))
    for anchor_ndx in range(len(gen_rgb_list)):
        # anchor_pos = poses[anchor_ndx]
        query = os.path.join(root_dir, "color", gen_rgb_list[anchor_ndx])
        anchor_pos = np.loadtxt(
            os.path.join(root_dir, "pose", gen_pose_list[anchor_ndx])
        )
        # Extract timestamp from the filename
        scan_filename = os.path.split(query)[1]
        timestamp = int(os.path.splitext(scan_filename)[0])

        positives = []
        non_negatives = []
        hard_ious = []
        hard_positives = []
        most_positive = [anchor_ndx]
        seq_ind = 0
        for i in range(len(cum_sum)):
            if anchor_ndx < cum_sum[i]:
                seq_ind = i
                break

        if seq_ind == 0:
            start = 0
            end = cum_sum[seq_ind]
        else:
            start = cum_sum[seq_ind - 1]
            end = cum_sum[seq_ind]

        if scene == "train":
            iou = iou_heaps[seq_ind]
        else:
            iou = test_iou_heaps[-1]

        for i in range(0, start):
            non_negatives.append(i)
        for i in range(end, cum_sum[-1]):
            non_negatives.append(i)

        max_ = 0

        for i in range(start, end):
            if i == anchor_ndx:
                non_negatives.append(i)
            if iou[anchor_ndx - start][i - start] > 0.3:
                positives.append(i)
                hard_ious.append(iou[anchor_ndx - start][i - start])
                if iou[anchor_ndx - start][i - start] > max_:
                    max_ = iou[anchor_ndx - start][i - start]
                    most_positive[0] = i
                non_negatives.append(i)
            elif iou[anchor_ndx - start][i - start] > 0:
                non_negatives.append(i)

        negatives = np.setdiff1d(labels, non_negatives, True)
        # if len(positives) != 0:
        #     argsort = np.argsort(-iou[positives])
        #     positives = list(np.array(positives)[argsort])
        #     most_positive.append(positives[0])
        #
        index = np.argsort(-np.array(hard_ious))
        hard_positives = np.array(positives)[index[:40]].tolist()
        positives = np.array(positives)
        non_negatives = np.array(non_negatives)

        if scene == "train":

            queries[anchor_ndx] = TrainingTuple(
                id=anchor_ndx,
                timestamp=timestamp,
                rel_scan_filepath=query,
                positives=positives,
                non_negatives=non_negatives,
                pose=anchor_pos,
            )
            file_path = os.path.join(root_dir, "kitti_train_tuple.pickle")
        else:

            queries[anchor_ndx] = TrainingTuple(
                id=anchor_ndx,
                timestamp=timestamp,
                rel_scan_filepath=query,
                positives=positives,
                non_negatives=non_negatives,
                pose=anchor_pos,
            )
            file_path = os.path.join(root_dir, "kitti_test_tuple.pickle")

    with open(file_path, "wb") as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)


gen_tuple("train")
gen_tuple("test")
