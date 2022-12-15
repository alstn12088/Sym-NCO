import math
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils import load_model, move_to
from utils.data_utils import save_dataset
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from utils.functions import parse_softmax_temperature

mp = torch.multiprocessing.get_context('spawn')


def SR_transform(x, y, idx):
    if idx < 0.5:
        phi = idx * 4 * math.pi
    else:
        phi = (idx - 0.5) * 4 * math.pi

    x = x - 1 / 2
    y = y - 1 / 2

    x_prime = torch.cos(phi) * x - torch.sin(phi) * y
    y_prime = torch.sin(phi) * x + torch.cos(phi) * y

    if idx < 0.5:
        dat = torch.cat((x_prime + 1 / 2, y_prime + 1 / 2), dim=2)
    else:
        dat = torch.cat((y_prime + 1 / 2, x_prime + 1 / 2), dim=2)
    return dat


def augment_xy_data_by_N_fold(problems, N, depot=None):
    x = problems[:, :, [0]]
    y = problems[:, :, [1]]

    if depot is not None:
        x_depot = depot[:, :, [0]]
        y_depot = depot[:, :, [1]]
    idx = torch.rand(N - 1)

    for i in range(N - 1):

        problems = torch.cat((problems, SR_transform(x, y, idx[i])), dim=0)
        if depot is not None:
            depot = torch.cat((depot, SR_transform(x_depot, y_depot, idx[i])), dim=0)

    if depot is not None:
        return problems, depot.view(-1, 2)

    return problems


def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems


def augment(input, N, problem):
    is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
    is_orienteering = problem.NAME == 'op'
    is_pctsp = problem.NAME == 'pctsp'
    if is_vrp or is_orienteering or is_pctsp:
        if is_vrp:
            features = ('demand',)
        elif is_orienteering:
            features = ('prize','max_length')
        else:
            assert is_pctsp
            features = ('deterministic_prize', 'penalty')

        input['loc'], input['depot'] = augment_xy_data_by_N_fold(input['loc'], N, depot=input['depot'].view(-1, 1, 2))

        for feat in features:
            if feat =='max_length':
                input[feat] = input[feat].repeat(N)
            else:
                input[feat] = input[feat].repeat(N, 1)
        
        return input

        # TSP
    return augment_xy_data_by_N_fold(input, N)


def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx + 1, ...], cost[idx:idx + 1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]


def eval_dataset_mp(args):
    (dataset_path, width, softmax_temp, opts, i, num_processes) = args

    model, _ = load_model(opts.model)
    val_size = opts.val_size // num_processes
    dataset = model.problem.make_dataset(filename=dataset_path, num_samples=val_size, offset=opts.offset + val_size * i)
    device = torch.device("cuda:{}".format(i))

    return _eval_dataset(model, dataset, width, softmax_temp, opts, device)


def eval_dataset(dataset_path, width, softmax_temp, opts):
    # Even with multiprocessing, we load the model here since it contains the name where to write results
    model, _ = load_model(opts.model)
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    if opts.multiprocessing:
        assert use_cuda, "Can only do multiprocessing with cuda"
        num_processes = torch.cuda.device_count()
        assert opts.val_size % num_processes == 0

        with mp.Pool(num_processes) as pool:
            results = list(itertools.chain.from_iterable(pool.map(
                eval_dataset_mp,
                [(dataset_path, width, softmax_temp, opts, i, num_processes) for i in range(num_processes)]
            )))

    else:
        device = torch.device("cuda:0" if use_cuda else "cpu")
        dataset = model.problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=opts.offset)
        costs, costs_augment, durations = _eval_dataset(model, dataset, width, softmax_temp, opts, device)

    # This is parallelism, even if we use multiprocessing (we report as if we did not use multiprocessing, e.g. 1 GPU)
    parallelism = opts.eval_batch_size

    # costs, costs_augment, durations = zip(*results)  # Not really costs since they should be negative

    print("Average cost:", costs.item())
    print("Average cost augment:", costs_augment.item())
    print("Average serial duration:", durations)

    return costs, costs_augment, durations


def _eval_dataset(model, dataset, width, softmax_temp, opts, device):
    model.to(device)
    model.eval()

    model.set_decode_type(
        "greedy")

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    results = []
    durations = 0
    cost_originals = 0
    cost_augments = 0
    i = 0
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        batch = move_to(batch, device)
        batch = augment(batch, opts.augment, model.problem)

        # batch = augment_xy_data_by_N_fold(batch,opts.augment)

        start = time.time()
        with torch.no_grad():
            costs, _ = model(batch)
            costs = costs.view(opts.augment, -1).permute(1, 0)
            cost_original = costs[:, 0]
            cost_augment, _ = costs.min(dim=1)

            batch_size = len(costs)
            ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)

        duration = time.time() - start
        durations += duration
        i += 1
        cost_original = cost_original.mean(dim=0)
        cost_augment = cost_augment.mean(dim=0)

        cost_originals += cost_original
        cost_augments += cost_augment

    return cost_originals / i, cost_augments / i, durations / i


if __name__ == "__main__":
    torch.manual_seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='data/pctsp/pctsp100_test_seed1234.pkl',
                        help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=50,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                        help="Softmax temperature (sampling or bs)")
    parser.add_argument('--model', type=str, default='pretrained_model/pctsp_100/epoch-99.pt')
    parser.add_argument('--augment', type=int, default=200)
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=10000, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Use multiprocessing to parallelize over multiple GPUs')

    opts = parser.parse_args()

    eval_dataset(opts.dataset_path, 0, opts.softmax_temperature, opts)