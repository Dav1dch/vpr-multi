import tqdm
import numpy as np
import torch
import torch.nn as nn
import pickle
import dgl
from gnn_loss import ERFA, sigmoid
import os
import argparse
from torch.optim.lr_scheduler import MultiStepLR


from gnn_utils import (
    load_minkLoc_model,
    make_dataloader,
    myGNN,
    myGNNCNN,
    get_latent_vectors,
)


def main():

    parser = argparse.ArgumentParser(description="Generate Baseline training dataset")
    parser.add_argument("--config", type=str, required=True, help="config file")
    parser.add_argument(
        "--model_config", type=str, required=True, help="model config file"
    )
    parser.add_argument("--weights", type=str, required=True, help="weights")
    parser.add_argument("--flag", type=int, required=True, help="generate or not")
    args = parser.parse_args()
    flag = args.flag

    # config = "/home/david/Code/S3E-rgb/config/config_baseline_multimodal.txt"
    # model_config = "/home/david/Code/S3E-rgb/models/minkloc3d.txt"
    # rgb_weights = "/home/david/datasets/weights_rgb/model_MinkLocRGB_20230517_000729_epoch_current_recall70.3_fire.pth"
    #
    # if torch.cuda.is_available():
    #     device = "cuda"
    # else:
    #     device = "cpu"
    # print("Device: {}".format(device))

    # # load minkloc
    mink_model, params = load_minkLoc_model(
        args.config, args.model_config, args.weights
    )
    mink_model.to("cuda")

    database_embeddings = []
    query_embeddings = []

    database_maps = []
    query_maps = []

    database_sets = None
    query_sets = None

    print(flag)
    for database_file, query_file in zip(
        params.eval_database_files, params.eval_query_files
    ):
        p = os.path.join(params.dataset_folder, database_file)
        with open(p, "rb") as f:
            database_sets = pickle.load(f)

        p = os.path.join(params.dataset_folder, query_file)
        with open(p, "rb") as f:
            query_sets = pickle.load(f)

        train_pose = []
        for set in database_sets:
            for ndx in tqdm.tqdm(set):
                R = np.loadtxt(
                    set[ndx]["query"]
                    .replace("color.png", "pose.txt")
                    .replace("color", "pose")
                )
                # train_pose.append(R)
                # rotations.append(quaternion.from_rotation_matrix(R[:3, :3]))
                train_pose.append(R[:3, -1])
        if flag:

            for set in tqdm.tqdm(database_sets):
                database_embeddings, database_maps = get_latent_vectors(
                    mink_model, set, "cuda", params
                )
                np.save("./gnn_pre_train_embeddings.npy", database_embeddings)
                np.save("./gnn_pre_train_maps.npy", database_maps)

            for set in tqdm.tqdm(query_sets):
                query_embeddings, query_maps = get_latent_vectors(
                    mink_model, set, "cuda", params
                )
                np.save("./gnn_pre_test_embeddings.npy", query_embeddings)
                np.save("./gnn_pre_test_maps.npy", query_maps)
            print(query_maps[0].shape)
    database_sets = database_sets[0]
    query_sets = query_sets[0]
    train_pose = torch.from_numpy(np.array(train_pose))

    iou = (1 / (torch.cdist(train_pose, train_pose) + 0.01)).type(dtype=torch.float32)

    def normalization(data):
        _range = torch.max(data) - torch.min(data)
        return (data - torch.min(data)) / _range

    iou = normalization(iou)
    print(iou)
    # iou = torch.tensor(
    #     np.load("/home/david/datasets/iou_/train_fire_iou.npy"), dtype=torch.float
    # )

    # embs = np.array(get_embeddings_3d(mink_model, params, "cuda", "train"))
    # np.save("./gnn_pre_train_embeddings.npy", embs)
    # test_embs = np.array(get_embeddings_3d(mink_model, params, "cuda", "test"))
    # np.save("./gnn_pre_test_embeddings.npy", test_embs)

    # load dataloaders
    dataloaders = make_dataloader(params)

    with open(os.path.join(params.dataset_folder, params.train_file), "rb") as f:
        train_pickle = pickle.load(f)

    model = myGNNCNN(256, 256, 128)
    model.to("cuda")

    opt = torch.optim.Adam(
        [{"params": model.parameters(), "lr": 0.0001, "weight_decay": 0.0001}]
    )
    scheduler = MultiStepLR(opt, milestones=[10], gamma=0.1)
    loss = None
    recall = None
    # smoothap = SmoothAP()
    # c2f = C2F()
    criterion = ERFA()
    d = {"loss": loss}

    database_maps = torch.tensor(np.load("./gnn_pre_train_maps.npy"))
    test_maps = torch.tensor(np.load("./gnn_pre_test_maps.npy"))

    embs = np.load("./gnn_pre_train_embeddings.npy")
    test_embs = np.load("./gnn_pre_test_embeddings.npy")
    # embs = np.hstack((embs, embs))
    # test_embs = np.hstack((test_embs, test_embs))
    database_embs = torch.tensor(embs.copy())
    query_embs = torch.tensor(test_embs.copy())
    # test_embs = torch.tensor(test_embs).to('cuda')
    database_embs = database_embs.to("cuda")
    query_embs = query_embs.to("cuda")

    embs = torch.tensor(embs).to("cuda")

    # temp = torch.repeat_interleave(embs_numpy, len(embs_numpy), 0) - torch.tile(embs_numpy, (len(embs_numpy), 1))
    criterion1 = nn.MSELoss().to("cuda")
    # shrinkage_loss = Shrinkage_loss(5, 0.2).to('cuda')
    pdist = nn.PairwiseDistance(p=2)
    cos = nn.CosineSimilarity(dim=1).cuda()

    max_ = 0.0
    # labels = range(len(feat))
    with tqdm.tqdm(range(100), position=0, desc="epoch", ncols=60) as tbar:
        for i in tbar:
            # loss status
            loss = 0.0
            losses = []
            cnt = 0.0
            num_evaluated = 0.0
            recall = [0] * 30
            with tqdm.tqdm(
                dataloaders["train"], position=1, desc="batch", ncols=80
            ) as tbar2:

                src = np.array(list(range(1, 31 * (31 - 1) + 1)))
                dst = np.repeat(list(range(31)), 31 - 1)

                g = dgl.graph((src, dst))
                g = g.to("cuda")

                for (
                    pos_mask,
                    neg_mask,
                    labels,
                    neighbours,
                    batch,
                ) in tbar2:
                    torch.cuda.empty_cache()
                    cnt += 1
                    model.train()
                    # mlpmodel.train()

                    with torch.enable_grad():
                        # batch = {e: batch[e].to(device) for e in batch}

                        # src = np.array(
                        #     list(range(1, len(labels) * (len(labels) - 1) + 1)))
                        # dst = np.repeat(list(range(len(labels))), len(labels) - 1)
                        # g = dgl.graph((src, dst))
                        # g = g.to('cuda')

                        ind = [labels[0]]
                        ind.extend(np.vstack(neighbours).reshape((-1,)).tolist())
                        indx = torch.tensor(ind).view((-1,))[dst[: len(labels) - 1]]
                        indy = torch.tensor(ind)[src[: len(labels) - 1]]
                        embeddings = embs[ind]
                        maps = database_maps[ind].cuda()
                        gt_iou = iou[indx, indy].view((-1, 1))
                        A, e = model(g, maps)
                        query_embeddings = torch.repeat_interleave(
                            A[0].unsqueeze(0), len(labels) - 1, 0
                        )
                        database_embeddings = A[1 : len(labels)]
                        sim_mat = cos(query_embeddings, database_embeddings)
                        # sim_mat = nn.functional.normalize(sim_mat, 2, 0)
                        d1 = database_embeddings.repeat(
                            1, len(database_embeddings), 1
                        ).squeeze()
                        d2 = database_embeddings.repeat_interleave(
                            len(database_embeddings), 0
                        )
                        database_sim_mat = cos(d1, d2).view(
                            (len(database_embeddings), len(database_embeddings))
                        )
                        # sim_mat[sim_mat < 0] = 0
                        # sim_mat = torch.matmul(query_embs, database_embs.T).squeeze()
                        loss_affinity_1 = criterion1(
                            e[: len(labels) - 1], gt_iou.cuda()
                        )

                        # hard_sim_mat = sim_mat[pos_mask.squeeze()[1:]]
                        # hard_pos_mask[0][0] = True
                        # hard_p_mask = hard_pos_mask[pos_mask].unsqueeze(0)
                        # ap_coarse = smoothap(sim_mat, pos_mask)
                        # ap_fine = smoothap(hard_sim_mat, hard_p_mask)

                        losses.append(
                            1 - criterion(sim_mat, pos_mask) + loss_affinity_1
                        )

                        # losses.append(
                        # c2f(sim_mat, database_sim_mat, pos_mask, hard_pos_mask,
                        #     neg_mask,  gt_iou_)
                        # 1
                        # - (0.7 * ap_coarse + 0.3 * ap_fine)
                        # + loss_affinity_1
                        # )

                        loss += losses[-1].item()
                        if cnt % 32 == 0 or cnt == len(database_sets):
                            a = torch.vstack(losses)
                            a = torch.where(torch.isnan(a), torch.full_like(a, 0), a)
                            loss_smoothap = torch.mean(a)
                            loss_smoothap.backward()
                            opt.step()
                            opt.zero_grad()
                            losses = []
                        #
                        rank = np.argsort(-sim_mat.detach().cpu().numpy())
                        # rank = rank[abs(rank - labels[0]) > 100]
                        # true_neighbors = database_sets[labels[0]][0]
                        true_neighbors = train_pickle[labels[0]].positives
                        if len(true_neighbors) == 0:
                            continue
                        num_evaluated += 1

                        flag = 0
                        for j in range(len(rank)):
                            if labels[1:][rank[j]] in true_neighbors:
                                recall[j - flag] += 1
                                break
                # tbar2.set_postfix({'loss': loss_smoothap.item()})
                print(loss / cnt)
                recall = (np.cumsum(recall) / float(num_evaluated)) * 100
                print("train recall\n", recall[:15])

                t_loss = 0.0
                scheduler.step()

                with torch.no_grad():
                    recall = [0] * 50
                    num_evaluated = 0
                    top1_similarity_score = []
                    one_percent_retrieved = 0
                    threshold = max(int(round(2000 / 100.0)), 1)

                    starter, ender = torch.cuda.Event(
                        enable_timing=True
                    ), torch.cuda.Event(enable_timing=True)
                    timings = np.zeros((6000, 1))

                    ndx = 0

                    model.eval()
                    t_loss = 0.0

                    src = np.array(list(range(1, 31 * (31 - 1) + 1)))
                    dst = np.repeat(list(range(31)), 31 - 1)

                    g = dgl.graph((src, dst))
                    g = g.to("cuda")
                    with tqdm.tqdm(
                        dataloaders["val"], position=1, desc="batch", ncols=80
                    ) as tbar3:
                        for (
                            pos_mask,
                            neg_mask,
                            labels,
                            neighbours,
                            batch,
                        ) in tbar3:
                            ind = [labels[0]]
                            # ind = labels
                            ind.extend(np.vstack(neighbours).reshape((-1,)).tolist())

                            embeddings = torch.vstack(
                                (query_embs[ind[0]], database_embs[ind[1:]])
                            )
                            maps = torch.vstack(
                                (test_maps[ind[0]].unsqueeze(0), database_maps[ind[1:]])
                            ).cuda()

                            starter.record()

                            A, e = model(g, maps)
                            ender.record()
                            torch.cuda.synchronize()  # 等待GPU任务完成
                            curr_time = starter.elapsed_time(ender)
                            timings[ndx] = curr_time
                            ndx += 1

                            database_embeddings = A[1 : len(labels)]

                            q = A[0].unsqueeze(0)

                            query_embeddings = torch.repeat_interleave(
                                q, len(labels) - 1, 0
                            )
                            # sim_mat = torch.matmul(q, database_embs.T).squeeze()

                            sim_mat = cos(query_embeddings, database_embeddings)

                            rank = torch.argsort((-sim_mat).squeeze())

                            true_neighbors = query_sets[labels[0]][0]
                            if len(true_neighbors) == 0:
                                continue
                            num_evaluated += 1

                            flag = 0
                            for j in range(len(rank)):
                                # if rank[j] == 0:
                                #     flag = 1
                                #     continue
                                if labels[1:][rank[j]] in true_neighbors:
                                    if j == 0:
                                        similarity = sim_mat[rank[j]]
                                        top1_similarity_score.append(similarity)
                                    recall[j - flag] += 1
                                    break

                            # if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
                            #     one_percent_retrieved += 1

                        # one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
                        recall = (np.cumsum(recall) / float(num_evaluated)) * 100
                        max_ = max(max_, recall[0])
                        print("recall\n", recall[:25])
                        # print(t_loss / num_evaluated)
                        print("max:", max_)
                        print("avg={}\n".format(timings.mean()))
                        # print(gt_iou.view(-1,)[:len(pos_mask[0])])

            tbar.set_postfix({"train loss": loss / cnt})


if __name__ == "__main__":
    main()
