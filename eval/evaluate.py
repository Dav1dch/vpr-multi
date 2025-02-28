# Author: Jacek Komorowski
# Warsaw University of Technology

# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad

from PIL import Image
from sklearn.neighbors import KDTree
import numpy as np
import pickle
import os
import argparse
import torch
import MinkowskiEngine as ME
import tqdm

from misc.utils import MinkLocParams
from models.model_factory import model_factory
from datasets.oxford import image4lidar
from datasets.augmentation import MinimizeTransform, TrainRGBTransform, ValRGBTransform

DEBUG = False


def evaluate(model, device, params, silent=True):
    # Run evaluation on all eval datasets
    assert len(params.eval_database_files) == len(params.eval_query_files)

    stats = {}
    for database_file, query_file in zip(
        params.eval_database_files, params.eval_query_files
    ):
        # Extract location name from query and database files
        location_name = database_file.split("_")[0]
        temp = query_file.split("_")[0]
        assert (
            location_name == temp
        ), "Database location: {} does not match query location: {}".format(
            database_file, query_file
        )

        p = os.path.join(params.dataset_folder, database_file)
        with open(p, "rb") as f:
            database_sets = pickle.load(f)

        p = os.path.join(params.dataset_folder, query_file)
        with open(p, "rb") as f:
            query_sets = pickle.load(f)

        temp = evaluate_dataset(
            model, device, params, database_sets, query_sets, silent=silent
        )
        stats[location_name] = temp

    return stats


def evaluate_dataset(model, device, params, database_sets, query_sets, silent=True):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    count = 0
    similarity = []
    one_percent_recall = []

    database_embeddings = []
    query_embeddings = []

    model.eval()

    for set in tqdm.tqdm(database_sets, disable=silent, ncols=70):
        database_embeddings.append(get_latent_vectors(model, set, device, params))

    for set in tqdm.tqdm(query_sets, disable=silent, ncols=70):
        query_embeddings.append(get_latent_vectors(model, set, device, params))

    # database_embeddings = query_embeddings

    for i in tqdm.tqdm(range(len(query_sets)), disable=silent):
        for j in range(len(query_sets)):
            # if i == j:
            #     continue
            pair_recall, pair_similarity, pair_opr = get_recall_(
                i, j, database_embeddings, query_embeddings, query_sets, database_sets
            )
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)

    ave_recall = recall / count
    average_similarity = np.mean(similarity)
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {
        "ave_one_percent_recall": ave_one_percent_recall,
        "ave_recall": ave_recall,
        "average_similarity": average_similarity,
    }
    return stats


def load_data_item(file_name, params):
    # returns Nx3 matrix
    file_path = os.path.join(params.dataset_folder, file_name).replace("color", "depth")

    result = {}
    # if params.use_cloud:
    #     pc = np.fromfile(file_path, dtype=np.float64)
    #     # coords are within -1..1 range in each dimension
    #     assert (
    #         pc.shape[0] == params.num_points * 3
    #     ), "Error in point cloud shape: {}".format(file_path)
    #     pc = np.reshape(pc, (pc.shape[0] // 3, 3))
    #     pc = torch.tensor(pc, dtype=torch.float)
    #     result["coords"] = pc
    #
    # if params.use_rgb:
    #     # Get the first closest image for each LiDAR scan
    #     assert os.path.exists(
    #         params.lidar2image_ndx_path
    #     ), f"Cannot find lidar2image_ndx pickle: {params.lidar2image_ndx_path}"
    #     lidar2image_ndx = pickle.load(open(params.lidar2image_ndx_path, "rb"))
    #     img = image4lidar(file_name, params.image_path, ".png", lidar2image_ndx, k=1)
    #     transform = ValRGBTransform()
    #     # Convert to tensor and normalize
    #     result["image"] = transform(img)

    img = Image.open(file_path)
    transform = MinimizeTransform()
    # result["image"] = transform(img)
    result = transform(img).to(torch.float)

    return result


import time


def get_latent_vectors(model, set, device, params):
    # Adapted from original PointNetVLAD code

    if DEBUG:
        embeddings = np.random.rand(len(set), 256)
        return embeddings

    model.eval()
    embeddings_l = []
    x = []
    times = []
    for elem_ndx in tqdm.tqdm(set, ncols=60):
        start = time.time()
        x = torch.stack(
            [
                load_data_item(
                    set[max(elem_ndx - i, (elem_ndx // 1000) * 1000)]["query"],
                    params,
                )
                for i in range(0, 5)
            ]
        )

        with torch.no_grad():
            # coords are (n_clouds, num_points, channels) tensor
            batch = {}

            if params.use_rgb:
                batch["images"] = x.unsqueeze(0).to(device)

            x = model(batch, False)
            embedding = x["embedding"]

            # embedding is (1, 256) tensor
            # if params.normalize_embeddings:
            #     embedding = torch.nn.functional.normalize(
            #         embedding, p=2, dim=1
            #     )  # Normalize embeddings

            end = time.time()
            times.append(end - start)
            embedding = embedding.detach().cpu().numpy()
            embeddings_l.append(embedding)
            x = []
    print(np.mean(times))

    embeddings = np.vstack(embeddings_l)
    return embeddings


def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets):
    # Original PointNetVLAD code
    database_output = database_vectors[m]
    queries_output = query_vectors[n]
    query = []
    reference = []

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        # {'query': path, 'northing': , 'easting': }
        query_details = query_sets[n][i]
        true_neighbors = query_details[m]

        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=100)
        true_indices = indices[0]
        to_remove = []
        for index, j in enumerate(indices[0]):
            if abs(i - j) <= 100 or j > i:
                to_remove.append(index)
        true_indices = np.delete(true_indices, to_remove)
        indices = np.array([true_indices])
        if len(indices) == 0:
            print("zero")
        query.append(i)
        reference.append(indices[0][0])

        for j in range(min(len(indices[0]), 25)):
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(
                        queries_output[i], database_output[indices[0][j]]
                    )
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if (
            len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors))))
            > 0
        ):
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100
    recall = (np.cumsum(recall) / float(num_evaluated)) * 100
    print(num_evaluated)
    # np.save('query.npy', np.array(query))
    # np.save('reference.npy', np.array(reference))
    return recall, top1_similarity_score, one_percent_recall


def get_recall_(m, n, database_vectors, query_vectors, query_sets, database_sets):
    # Original PointNetVLAD code
    database_output = database_vectors[m]
    queries_output = query_vectors[n]

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[n][i]  # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(
            np.array([queries_output[i]]), k=num_neighbors
        )

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(
                        queries_output[i], database_output[indices[0][j]]
                    )
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if (
            len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors))))
            > 0
        ):
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100
    recall = (np.cumsum(recall) / float(num_evaluated)) * 100
    return recall, top1_similarity_score, one_percent_recall


def print_eval_stats(stats):
    for database_name in stats:
        print("Dataset: {}".format(database_name))
        t = "Avg. top 1% recall: {:.2f}   Avg. similarity: {:.4f}   Avg. recall @N:"
        print(
            t.format(
                stats[database_name]["ave_one_percent_recall"],
                stats[database_name]["average_similarity"],
            )
        )
        print(stats[database_name]["ave_recall"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on RobotCar dataset")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to the model-specific configuration file",
    )
    parser.add_argument(
        "--weights", type=str, required=False, help="Trained model weights"
    )

    args = parser.parse_args()
    print("Config path: {}".format(args.config))
    print("Model config path: {}".format(args.model_config))
    if args.weights is None:
        w = "RANDOM WEIGHTS"
    else:
        w = args.weights
    print("Weights: {}".format(w))
    print("")

    params = MinkLocParams(args.config, args.model_config)
    params.print()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Device: {}".format(device))

    model = model_factory(params)
    if args.weights is not None:
        assert os.path.exists(args.weights), "Cannot open network weights: {}".format(
            args.weights
        )
        print("Loading weights: {}".format(args.weights))
        model.load_state_dict(torch.load(args.weights, map_location=device))

    model.to(device)

    stats = evaluate(model, device, params, silent=False)
    print_eval_stats(stats)
