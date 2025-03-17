import numpy as np
import torch
from tqdm import tqdm
from index import Index
from argparse import ArgumentParser
from pathlib import Path
import numpy as np


def evaluate(
    index: Index,
) -> (float, np.ndarray):
    """
    Evaluates a pre-trained model's mean Average Precision (mAP) and Rank-N accuracy
    from query results on the gallery set. It is assumed that query embeddings and meta-data
    are included in the index.

    Args:
        index: Index object with gallery and query data
    Returns:
        mAP: mean Average Precision
        rank_n: Rank-N accuracy array where each position denotes the model's retreival accuracy at that rank
    """

    n_queries = index.query_cls_labels.shape[1]
    n_gallery = index.gallery_cls_labels.shape[1]

    ap = 0
    rank_n = torch.zeros(n_gallery, dtype=torch.int)

    gl = index.gallery_cls_labels
    gc = index.gallery_cam_ids

    # use pre-computed query embeddings to search the gallery and compute performance metrics
    for i in tqdm(range(n_queries)):
        qf = index.query_embs[i].reshape(1, -1)
        ql = index.query_cls_labels[0, i]
        qc = index.query_cam_ids[0, i]

        _, indexes = index.search(qf, topk=gl.shape[1])

        indexes = indexes.squeeze()
        query_index = np.argwhere(gl == ql)
        camera_index = np.argwhere(gc == qc)
        gold_index = np.setdiff1d(query_index, camera_index, assume_unique=True)

        # remove distractor and identical images
        distractor_index = np.argwhere(gl == -1)
        identical_index = np.intersect1d(query_index, camera_index)
        ignore_index = np.append(identical_index, distractor_index)

        query_ap = 0
        query_rank = torch.zeros(len(indexes), dtype=torch.int)

        # remove irrelevant images
        mask = np.isin(indexes, ignore_index, invert=True)
        indexes = indexes[mask]

        # evaluate AP of suitable indexes
        n_gold = len(gold_index)
        mask = np.isin(indexes, gold_index)
        matches = np.argwhere(mask == True).flatten()

        # value of match denotes retrieval rank
        query_rank[matches] = 1
        d_recall = 1 / n_gold

        for i in range(n_gold):
            precision = (i + 1) * 1.0 / (matches[i] + 1)

            if matches[i] != 0:
                old_precision = i * 1.0 / matches[i]
            else:
                old_precision = 1.0

            query_ap += d_recall * (old_precision + precision) / 2

        ap += query_ap
        rank_n += query_rank

    rank_n = rank_n / n_queries
    mAP = ap / n_queries

    return mAP, rank_n


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Evaluates the model on gallery and query images."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="masked-hog-908",
        help="Experiment name. Example: masked-hog-908",
    )
    parser.add_argument(
        "--load_index",
        action="store_true",
        default=False,
        help="If set, a serialized index is loaded from disk.",
    )
    parser.add_argument(
        "--save_index",
        action="store_true",
        default=False,
        help="If set, the built index is serialized to disk.",
    )
    args = parser.parse_args()

    model_path = Path(f"./models/{args.exp_name}") / "best.pth"

    # create Index with query embeddings for evaluation
    index = Index(
        model_path=model_path,
        index_name=args.exp_name,
        load_index=args.load_index,
        save_index=args.save_index,
        query=True,
    )

    print("Evaluating Query images on Gallery index...")
    mAP, rank_n = evaluate(index)

    top1 = rank_n[0] * 100
    top5 = rank_n[4] * 100
    top10 = rank_n[9] * 100

    print(
        f"Top-1 accuracy: {top1:.2f}%, Top-5 accuracy: {top5:.2f}%, Top-10 accuracy: {top10:.2f}%, mAP: {mAP*100:.2f}%"
    )
