import os
import shutil
import argparse
import time
from typing import Optional, Callable

from ogb.nodeproppred import PygNodePropPredDataset
import torch
from torch_geometric.datasets import Planetoid, WikiCS, Amazon, Reddit
from torch_geometric.data.data import BaseData
from torch_sparse import SparseTensor

# fix for https://github.com/snap-stanford/ogb/issues/497
###
import torch
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])
####

def safe_download_dataset(
    download_func: Callable[[], BaseData],
    dataset_name: str,
    max_retries: int = 3
) -> Optional[BaseData]:
    """Safely download a dataset with error handling and retries."""
    for attempt in range(max_retries):
        try:
            print(f"\nAttempting to download {dataset_name} (Attempt {attempt + 1}/{max_retries})")
            return download_func()
        except Exception as e:
            print(f"Error downloading {dataset_name}: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                if os.path.exists('temp_download'):
                    shutil.rmtree('temp_download')
                os.makedirs("temp_download", exist_ok=True)
            else:
                print(f"Failed to download {dataset_name} after {max_retries} attempts")
    return None


def convert_to_index_tensors(mask: torch.Tensor) -> torch.Tensor:
    """Convert boolean mask or index tensor to long index tensor."""
    if mask.dtype == torch.bool:
        return mask.nonzero(as_tuple=False).view(-1)
    return mask.long() if mask.dtype != torch.long else mask


def process_dataset(
    data: BaseData,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    num_classes: int
) -> BaseData:
    """Process dataset into consistent format with index-based masks."""
    data.train_masks = [convert_to_index_tensors(train_mask)]
    data.val_masks = [convert_to_index_tensors(val_mask)]
    data.test_masks = [convert_to_index_tensors(test_mask)]
    data.label_names = [str(i) for i in range(num_classes)]

    # Ensure labels are 1D class indices
    if hasattr(data, 'y') and data.y is not None:
        data.y = data.y.squeeze()
        if data.y.dim() > 1:
            data.y = data.y.argmax(dim=1)

    print(f"\nProcessed dataset - Train: {len(data.train_masks[0])}, "
          f"Val: {len(data.val_masks[0])}, Test: {len(data.test_masks[0])}")

    return data


def make_symmetric_sparse_tensor(edge_index: torch.Tensor, num_nodes: int) -> SparseTensor:
    """Create symmetric sparse tensor from edge index."""
    sparse = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes))
    return sparse.to_symmetric()


def download_ogbn_arxiv(save_path: str) -> BaseData:
    """Download and process OGBN-Arxiv dataset."""
    print("\nDownloading OGBN-Arxiv dataset...")
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='temp_download')
    data: BaseData = dataset[0]

    num_nodes: int = data.num_nodes if data.num_nodes is not None else 0
    data.edge_index = make_symmetric_sparse_tensor(data.edge_index, num_nodes)
    splits = dataset.get_idx_split()

    processed_data = process_dataset(
        data, splits['train'], splits['valid'], splits['test'], dataset.num_classes
    )

    output_path = os.path.join(save_path, 'ogbn_arxiv.pt')
    torch.save(processed_data, output_path)
    print(f"Saved to: {output_path}")
    return processed_data


def download_planetoid(name: str, save_path: str) -> BaseData:
    """Download and process Planetoid dataset (Cora, CiteSeer, PubMed)."""
    print(f"\nDownloading {name} dataset...")
    dataset = Planetoid(root=f'/tmp/{name}', name=name)
    data: BaseData = dataset[0]

    num_nodes: int = data.num_nodes if data.num_nodes is not None else 0
    data.edge_index = make_symmetric_sparse_tensor(data.edge_index, num_nodes)

    processed_data = process_dataset(
        data, data.train_mask, data.val_mask, data.test_mask, dataset.num_classes
    )

    output_path = os.path.join(save_path, f'{name.lower()}.pt')
    torch.save(processed_data, output_path)
    print(f"Saved to: {output_path}")
    return processed_data


def download_wikics(save_path: str) -> BaseData:
    """Download and process WikiCS dataset with multiple splits."""
    print("\nDownloading WikiCS dataset...")
    dataset = WikiCS(root='temp_download')
    data = dataset[0]

    data.edge_index = make_symmetric_sparse_tensor(data.edge_index, data.num_nodes)

    # WikiCS has multiple train/val splits but single test split
    data.train_masks = [
        data.train_mask[:, i].nonzero(as_tuple=False).view(-1)
        for i in range(data.train_mask.size(1))
    ]
    data.val_masks = [
        data.val_mask[:, i].nonzero(as_tuple=False).view(-1)
        for i in range(data.val_mask.size(1))
    ]
    data.test_masks = [data.test_mask.nonzero(as_tuple=False).view(-1)]
    data.label_names = [str(i) for i in range(dataset.num_classes)]

    output_path = os.path.join(save_path, 'wikics.pt')
    torch.save(data, output_path)
    print(f"Saved to: {output_path}")
    return data


def download_amazon(name: str, save_path: str) -> BaseData:
    """Download and process Amazon dataset (Photo or Computers)."""
    print(f"\nDownloading Amazon-{name} dataset...")
    dataset = Amazon(root='temp_download', name=name)
    data: BaseData = dataset[0]

    num_nodes: int = data.num_nodes if data.num_nodes is not None else 0
    data.edge_index = make_symmetric_sparse_tensor(data.edge_index, num_nodes)

    # Create random 60/20/20 train/val/test split
    indices = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)

    processed_data = process_dataset(
        data,
        train_mask=indices[:train_size],
        val_mask=indices[train_size:train_size + val_size],
        test_mask=indices[train_size + val_size:],
        num_classes=dataset.num_classes
    )

    output_path = os.path.join(save_path, f'amazon_{name.lower()}.pt')
    torch.save(processed_data, output_path)
    print(f"Saved to: {output_path}")
    return processed_data


def download_reddit(save_path: str) -> BaseData:
    """Download and process Reddit dataset."""
    print("\nDownloading Reddit dataset...")
    dataset = Reddit(root='temp_download')
    data: BaseData = dataset[0]
    num_nodes: int = data.num_nodes if data.num_nodes is not None else 0

    data.edge_index = make_symmetric_sparse_tensor(data.edge_index, num_nodes)

    processed_data = process_dataset(
        data, data.train_mask, data.val_mask, data.test_mask, dataset.num_classes
    )

    output_path = os.path.join(save_path, 'reddit.pt')
    torch.save(processed_data, output_path)
    print(f"Saved to: {output_path}")
    return processed_data


def print_dataset_info(data: BaseData, name: str) -> None:
    """Print summary statistics for a dataset."""
    print(f"\n{name} Dataset Info:")
    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.edge_index.nnz():,}")
    print(f"  Features: {data.x.size(1)}")
    print(f"  Classes: {len(data.label_names)}")
    print(f"  Train: {len(data.train_masks[0]):,}")
    print(f"  Val: {len(data.val_masks[0]):,}")
    print(f"  Test: {len(data.test_masks[0]):,}")


def main():
    """Main function to download and process graph datasets."""
    print("Starting dataset downloader...")

    parser = argparse.ArgumentParser(description='Download and process graph datasets')
    parser.add_argument(
        '--save-path',
        type=str,
        default='../datasets',
        help='Path to save the processed datasets'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['arxiv', 'cora', 'wikics', 'amazon-photo', 'amazon-computers', 'reddit'],
        help='Name of the dataset to download'
    )

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs("temp_download", exist_ok=True)

    # Dataset download mapping
    download_funcs = {
        'arxiv': lambda: download_ogbn_arxiv(args.save_path),
        'cora': lambda: download_planetoid('Cora', args.save_path),
        'wikics': lambda: download_wikics(args.save_path),
        'amazon-photo': lambda: download_amazon('Photo', args.save_path),
        'amazon-computers': lambda: download_amazon('Computers', args.save_path),
        'reddit': lambda: download_reddit(args.save_path),
    }

    try:
        data = safe_download_dataset(download_funcs[args.dataset], args.dataset)
        if data is not None:
            print_dataset_info(data, args.dataset)
        else:
            print("\nDownload failed!")
    finally:
        # Cleanup temporary directory
        print("Cleaning up temporary files...")
        if os.path.exists('temp_download'):
            shutil.rmtree('temp_download')

if __name__ == "__main__":
    main()
