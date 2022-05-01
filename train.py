import argparse
import os.path

import pandas as pd
import torch.cuda
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import FlyData, TaskSampler, prepare_nshot_task
from model import prototype_encoder


def proto_net_episode(model, optimiser, loss_fn, x, y, n_shot, k_way, distance, train):
    if train:
        model.train()
        optimiser.zero_grad()
    else:
        model.eval()

    embeddings = model(x)

    support = embeddings[:n_shot * k_way]
    query = embeddings[n_shot * k_way:]
    prototypes = support.reshape(k_way, n_shot, -1).mean(dim=1)
    distance = pairwise_distances(query, prototypes, distance)

    log_p_y = (-distance).log_softmax(dim=1)
    loss = loss_fn(log_p_y, y)

    y_pred = (-distance).softmax(dim=1)

    if train:
        loss.backward()
        optimiser.step()
    else:
        pass

    return loss, y_pred


def pairwise_distances(x, y, matching_fn):
    EPSILON = 1e-8
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances

    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise (ValueError('Unsupported similarity function'))


def categorical_accuracy(y, y_pred):
    return torch.eq(y_pred.argmax(dim=-1), y).sum().item() / y_pred.shape[0]


NAMED_METRICS = {
    'categorical_accuracy': categorical_accuracy
}

with torch.no_grad():
    def batch_metrics(model, y_pred, y, metrics, batch_logs):
        model.eval()
        for m in metrics:
            if isinstance(m, str):
                batch_logs[m] = NAMED_METRICS[m](y, y_pred)
            else:
                # Assume metric is a callable function
                batch_logs = m(y, y_pred)

        return batch_logs


def val_fun(taskloader, model, optimiser, loss_fun, n_shot, k_way, q_queries, distance):
    seen = 0
    totals = dict(loss=0.0, val_acc=0.0)
    logs = dict(loss=0.0, val_acc=0.0)
    print('evaluating...')
    # val_bar = tqdm(total=len(taskloader), desc='Epoch {}'.format(epoch))
    with torch.no_grad():
        for batch_index, batch in enumerate(taskloader):
            x, y = prepare_nshot_task(batch, k_way, q_queries)
            loss, y_pred = proto_net_episode(model, optimiser, loss_fun, x, y, n_shot, k_way, distance, train=False)
            seen += y_pred.shape[0]

            totals['loss'] += loss.item() * y_pred.shape[0]
            totals['val_acc'] += categorical_accuracy(y, y_pred) * y_pred.shape[0]
            # totals['loss'] = totals['loss'] / seen
            # totals['val_acc'] = totals['val_acc'] / seen
            # val_bar.update(1)
            # val_bar.set_postfix(totals)
        logs['val_loss'] = totals['loss'] / seen
        logs['val_acc'] = totals['val_acc'] / seen
    print(f'evaluated, val_acc:{logs["val_acc"]}, val_loss:{logs["val_loss"]}')
    return logs


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-train', default='data/dataset1/train')
    parser.add_argument('--dataset-valid', default='data/dataset1/valid')
    parser.add_argument('--distance', default='l2')
    parser.add_argument('--n-train', default=1, type=int)
    parser.add_argument('--n-test', default=1, type=int)
    parser.add_argument('--k-train', default=10, type=int)
    parser.add_argument('--k-test', default=5, type=int)
    parser.add_argument('--q-train', default=5, type=int)
    parser.add_argument('--q-test', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--epoch_num', default=1, type=int)
    parser.add_argument('--logs_folder', default='logs', type=str)
    parser.add_argument('--model_folder', default='model', type=str)

    args = parser.parse_args()
    return args


def save_logs(logs, args):
    logs_csv = pd.DataFrame(logs)
    if not os.path.exists(args.logs_folder):
        os.mkdir(args.logs_folder)
    dataset_name = args.dataset_train.split('/')[-2]
    logs_csv.to_csv(f'{args.logs_folder}/{dataset_name}_{args.n_train}shot_{args.k_train}way_log.csv')


def save_model(model, args, type='best'):
    if not os.path.exists(args.model_folder):
        os.mkdir(args.model_folder)
    torch.save(model,f'{args.model_folder}/{args.n_train}shot_{args.k_train}way_{type}.pth')

def run_prototype(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = FlyData(phase='train', data_root=f'{args.dataset_train}')
    val_dataset = FlyData(phase='valid', data_root=f'{args.dataset_valid}')
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=TaskSampler(train_dataset, episodes_per_epoch=100,
                                  n=args.n_train, k=args.k_train, q=args.q_train),
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=TaskSampler(val_dataset, episodes_per_epoch=1000,
                                  n=args.n_test, k=args.k_test, q=args.q_test),
        num_workers=args.num_workers
    )

    model = prototype_encoder(3)
    model = model.to(device, dtype=torch.double)
    optimiser = Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.NLLLoss().cuda()
    metrics = ['categorical_accuracy']

    epoch_num = args.epoch_num
    logs = []
    last_acc = 0.0
    for epoch in range(1, epoch_num + 1):
        batch_log_bar = tqdm(total=len(train_loader), desc='Epoch {} / {}'.format(epoch, epoch_num), position=0)
        batch_logs = dict()
        for batch_index, batch in enumerate(train_loader):
            x, y = prepare_nshot_task(batch, k=args.k_train, q=args.q_train)
            loss, y_pred = proto_net_episode(model, optimiser, loss_fn, x, y,
                                             args.n_train, args.k_train, 'l2', train=True)
            batch_logs = batch_metrics(model, y_pred, y, metrics, batch_logs)

            batch_logs['loss'] = loss.item()
            batch_log_bar.set_postfix(batch_logs)
            batch_log_bar.update(1)
        batch_log_bar.close()

        log = val_fun(val_loader, model, optimiser, loss_fn,
                      args.n_test, args.k_test, args.q_test, 'l2')
        log['epoch'] = epoch
        logs.append(log)
        save_logs(logs, args)

        if log['val_acc'] > last_acc:
            last_acc = log['val_acc']
            save_model(model, args, type='best')
        save_model(model, args, type='last')


def main():
    args = parse_opt()
    run_prototype(args)


if __name__ == "__main__":
    main()
