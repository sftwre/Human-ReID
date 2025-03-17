from pathlib import Path
from torchvision import datasets
from datasets import transforms_minimal as transform
import torch.utils.data as data
import numpy as np
import scipy.io as io
import os
from model import ImgEmbedder
import faiss
from PIL import Image
from typing import List
import numpy as np


def get_meta_data(file_paths: List[tuple[str, int]]) -> (np.ndarray, np.ndarray):
    """
    Extracts camera id and class label from dataset file paths.

    Args:
        file_paths: list of file paths within the Market-1501 dataset

    Returns:
        numpy array of camera ids and class labels parsed from file paths
    """

    camera_ids = []
    classes = []

    for path, _ in file_paths:

        person_id, cam_id, frame_id, bbox_id = path.split("/")[-1].split("_")
        camera_ids.append(int(cam_id[1]))
        classes.append(int(person_id))

    camera_ids = np.array(camera_ids).reshape(1, -1)
    classes = np.array(classes).reshape(1, -1)

    return camera_ids, classes


class Index:

    def __init__(
        self,
        model_path: str,
        index_name: str,
        load_index: bool = False,
        save_index: bool = False,
        query: bool = False,
    ):
        """
        Builds a FAISS index on the gallery image-embeddings to facilitate retrieval.
        If the path to a serialized index is provided, the index is loaded from disk.
        Otherwise, the index is built from the gallery and query datasets.

        Args:
            model_path: Path to pre-trained model weights
            index_name: Name of index. This is used as the file name if save_index is True.
            load_index: Flag used to load a serialied index from disk
            save_index: Flad used to serialize an index to disk
            query: Flag to control query-embedding extraction. Recommended for evaluation
        """

        # singleton attribute that references a FAISS index
        self.index = None

        self.model = ImgEmbedder(model_path=model_path)
        self.index_path = Path(f"./index/{index_name}.mat")
        self.data_path = Path("./data/market-1501-grouped")

        self.gallery_dataset = datasets.ImageFolder(
            root=self.data_path / "gallery", transform=transform
        )

        self.gallery_dataloader = data.DataLoader(
            self.gallery_dataset,
            batch_size=16,
            shuffle=False,
        )

        self.query_dataset = self.query_dataloader = None

        self.query = query

        # load query dataset if query is set to true
        if self.query:

            self.query_dataset = datasets.ImageFolder(
                root=self.data_path / "query", transform=transform
            )

            self.query_dataloader = data.DataLoader(
                self.query_dataset,
                batch_size=16,
                shuffle=False,
            )

        if load_index:
            self._load_index()
        else:
            self._build_index()

        if save_index:
            self._write_index()

    def _init_index(self, embs: np.ndarray):
        """
        Initializes the FAISS index with the provided embeddings
        and assigns it to the instance attribute `index`, which is a singleton object.

        Args:
            embs: Normalized image embeddings
        """

        if self.index is None:
            print("Initializing search index...")
            self.index = faiss.IndexFlatIP(self.model.emb_dim)
            self.index.add(embs)
            print("Done!")

    def _build_index(self):
        """
        Extracts image embeddings and meta-data from the gallery dataset
        and initializes the FAISS index with the embeddings.
        If query flag is True, then query embeddings and meta-data are also extracted.
        """

        self._set_gallery_attr()

        # query embeddings and meta-data not extracted by default.
        self.query_embs = []
        self.query_cam_ids = []
        self.query_cls_labels = []

        if self.query:
            self._set_query_attr()

        self._init_index(self.gallery_embs)

    def _load_index(self):
        """
        Loads a serialized index from disk and sets instance attributes.
        """

        if not self.index_path.exists():
            raise IOError(f"Index file not found at: {self.index_path}")

        print(f"Loading serialized index from {self.index_path}...")
        index = io.loadmat(self.index_path)

        self.gallery_embs = index["gallery_embs"]
        self.gallery_cam_ids = index["gallery_cam_ids"]
        self.gallery_cls_labels = index["gallery_cls_labels"]

        self.query_embs = index["query_embs"]
        self.query_cam_ids = index["query_cam_ids"]
        self.query_cls_labels = index["query_cls_labels"]

        # if no query data was serialized but was requested, extract query embeddings
        if (
            not (
                self.query_embs.size
                and self.query_cam_ids.size
                and self.query_cls_labels.size
            )
            and self.query
        ):
            self._set_query_attr()

        self._init_index(self.gallery_embs)

    def _write_index(self):
        """
        Saves the index to disk within the ./index/ directory.
        """

        if not self.index_path.parent.exists():
            os.mkdir(self.index_path.parent)

        print("Saving index to disk...")

        io.savemat(
            self.index_path,
            {
                "query_embs": self.query_embs,
                "query_cam_ids": self.query_cam_ids,
                "query_cls_labels": self.query_cls_labels,
                "gallery_embs": self.gallery_embs,
                "gallery_cam_ids": self.gallery_cam_ids,
                "gallery_cls_labels": self.gallery_cls_labels,
            },
        )

        print(f"Index saved to: {self.index_path}")

    def _set_gallery_attr(self):
        """
        Sets instance attributes related to gallery dataset.
        """

        print("Extracting Gallery embeddings...")
        self.gallery_embs = self.model.to_embeddings(self.gallery_dataloader).numpy()
        self.gallery_cam_ids, self.gallery_cls_labels = get_meta_data(
            self.gallery_dataset.imgs
        )

    def _set_query_attr(self):
        """
        Sets instance attributes related to query dataset. Called when query flag is True.
        """

        print("Extracting Query embeddings...")
        self.query_embs = self.model.to_embeddings(self.query_dataloader).numpy()
        self.query_cam_ids, self.query_cls_labels = get_meta_data(
            self.query_dataset.imgs
        )

    def search(
        self, query: np.ndarray | str, topk: int = 5
    ) -> (np.ndarray, np.ndarray):
        """
        Search the gallery for similar images to the query.
        If a query image path is provided, the image is loaded and its embeddings are computed.
        Otherwise, the query embeddings are used directly.

        Args:
            query: Path to query image or query image embeddings
            topk: Number of top similar images to return
        Returns:
            scores: Cosine similarity scores
            indices: Index of similar images within the gallery
        """

        # load query image
        if isinstance(query, str):
            query_img = Image.open(query).convert("RGB")
            query_tensor = transform(query_img)
            query_tensor = query_tensor.unsqueeze(0)
            query = self.model(query_tensor).reshape(1, -1).numpy()

            # normalize to unit length
            query = query / np.linalg.norm(query, axis=1, keepdims=True)

        query_emb = query
        scores, indices = self.index.search(query_emb, k=topk)

        return scores, indices
