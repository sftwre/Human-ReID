from pathlib import Path
from argparse import ArgumentParser
from index import Index

if __name__ == "__main__":

    parser = ArgumentParser(description="Performs inference on a trained model.")
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
    parser.add_argument(
        "--query_paths",
        type=str,
        nargs="+",
        help="List of query image paths",
        default="./data/market-1501-grouped/query/0075/0075_c2s1_010651_00.jpg",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Path to directory with trained model weights. Example: masked-hog-908",
        default="masked-hog-908",
    )
    parser.add_argument(
        "--topk", default=5, type=int, help="Number of top similar images to return"
    )
    args = parser.parse_args()

    query_img_paths = args.query_paths

    if isinstance(query_img_paths, str):
        query_img_paths = list([query_img_paths])

    try:

        # create model path from experiment name
        model_dir = Path("./models") / args.exp_name
        model_path = model_dir / "best.pth"

        index = Index(
            model_path=model_path,
            index_name=args.exp_name,
            load_index=args.load_index,
            save_index=args.save_index,
        )

        for query_path in query_img_paths:

            scores, indexes = index.search(query_path, topk=args.topk)

            gallery_paths = index.gallery_dataset.imgs

            print(f"Query image: {query_path}")
            print(f"Top {args.topk} similar images:")

            for i, idx in enumerate(indexes[0]):
                result_path = str(gallery_paths[idx.item()][0]).strip()
                print(f"{i+1}: {result_path}, Score: {scores[0][i]:.4f}")
            print("")

    except Exception as e:
        print(e)
