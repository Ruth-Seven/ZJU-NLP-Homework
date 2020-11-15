import argparse
from pathlib import Path

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

from train import train
import utils


def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')
    utils.parse_optimizer(parser)

    parser.add_argument('--model_type', type=str,
                        help='Type of GNN model.')
    parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    parser.add_argument('--num_layers', type=int,
                        help='Number of graph conv layers')
    parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--dataset', type=str,
                        help='Dataset')

    parser.set_defaults(model_type='GAT',
                        dataset='cora',
                        num_layers=2,
                        batch_size=64,
                        hidden_dim=32,
                        dropout=0.5,
                        epochs=200,
                        opt='adam',
                        opt_scheduler='none',
                        weight_decay=0.0,
                        lr=0.01)

    return parser.parse_args()


def main():
    args = arg_parse()
    path = Path("../data")
    if args.dataset == 'cora':
        dataset = Planetoid(root=path / 'Cora', name='Cora', split='random', num_train_per_class=77)
        task = 'node'
    elif args.dataset == 'citeseer':
        dataset = Planetoid(root=path / 'CiteSeer', name='CiteSeer', split='random', num_train_per_class=111)
        task = 'node'
    # print(dataset.data)
    # print(dataset.num_classes)
    return train(dataset, task, args)


if __name__ == '__main__':
    main()

