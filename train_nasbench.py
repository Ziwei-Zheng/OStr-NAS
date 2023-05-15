import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from nasbench2 import NAS201Model as Network, get_arch_str
from architect_nasbench import Architect
from nas_201_api import NASBench201API as API
import random
from imagenet16 import ImageNet16


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/workspace/data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', '--b', type=int, default=512, help='batch size')
parser.add_argument('--learning_rate', '--lr', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=150, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--save', type=str, default='', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--single_level', action='store_true', default=False, help='use single level optimization')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')


args = parser.parse_args()
if args.save == '':
    args.save = 'search_nasbench/b{}_{}'.format(args.batch_size, args.seed)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.log'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.set == 'cifar10':
    CLASSES = 10
elif args.set == 'cifar100':
    CLASSES = 100
elif args.set == 'ImageNet16-120':
    CLASSES = 120

def set_seed(seed):
    np.random.seed(seed)
    cudnn.benchmark = False
    torch.manual_seed(seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    
    torch.cuda.set_device(args.gpu)
    set_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(CLASSES, stem_ch=args.init_channels, layers=args.layers).cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    import itertools
    optimizer = torch.optim.SGD(
        model.parameters() if not args.single_level else itertools.chain(model.parameters(), [model.alphas]),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    arch_optimizer = torch.optim.Adam([model.alphas],
        lr=args.arch_learning_rate, betas=(0.9, 0.999), weight_decay=args.arch_weight_decay)

    if args.set == 'cifar10':
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.set == 'cifar100':
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        test_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.set == 'ImageNet16-120':
        train_transform, valid_transform = utils._data_transforms_imagenet16(args)
        train_data = ImageNet16(root=args.data + '/ImageNet16', train=True, transform=train_transform, use_num_of_class_only=120)
        test_data = ImageNet16(root=args.data + '/ImageNet16', train=False, transform=valid_transform, use_num_of_class_only=120)

    if not args.single_level:
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))
        train_dataset, valid_dataset = torch.utils.data.random_split(train_data, [split, num_train-split])
        train_queue = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        valid_queue = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    else:
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            pin_memory=True, num_workers=4)
        valid_queue = None

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size,
        pin_memory=True, num_workers=4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, criterion, arch_optimizer, args)

    logging.info('loading nas_bench_api...')
    api = API('../../data/NAS-Bench-102-v1_0-e61699.pth', verbose=False)
    accs, rks = get_nas_bench(api, args.set)
    logging.info('nas_bench_api loaded')

    best_acc = 0.0
    for epoch in range(args.epochs):

        lr = scheduler.get_last_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        # training
        train_acc, train_obj, alpha_grad_sum, weight_grad_sum, zz_grad_sum = train(
            train_queue, valid_queue, model, architect, criterion, optimizer, lr, arch_optimizer)
        logging.info('train_acc %f', train_acc)

        alpha = model.alphas
        arch = get_arch_str(alpha, False)
        index = api.query_index_by_arch(arch)
        logging.info('  A : arch %s, index %d, test_acc %2f, ranking %d', arch, index, accs[index], rks[index])

        arch = get_arch_str(alpha_grad_sum, False)
        index = api.query_index_by_arch(arch)
        logging.info('|gA|: arch %s, index %d, test_acc %2f, ranking %d', arch, index, accs[index], rks[index])

        arch = get_arch_str(weight_grad_sum, False)
        index = api.query_index_by_arch(arch)
        logging.info('|gW|: arch %s, index %d, test_acc %2f, ranking %d', arch, index, accs[index], rks[index])

        arch = get_arch_str(zz_grad_sum, False)
        index = api.query_index_by_arch(arch)
        logging.info('|gZ|: arch %s, index %d, test_acc %2f, ranking %d', arch, index, accs[index], rks[index])

        scheduler.step()

        if not args.single_level:
            # validation
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)

        # testing
        test_acc, test_obj = infer(test_queue, model, criterion, test=True)
        if test_acc > best_acc:
            best_acc = test_acc
        logging.info('test_acc %f, best_acc %f', test_acc, best_acc)

        torch.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, arch_optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    Loss = utils.AvgrageMeter()

    arch_grads_sum = torch.zeros_like(model.alphas).cuda()
    weight_grads_sum = torch.zeros_like(model.weights).cuda()
    zz_grads_sum = torch.zeros_like(model.c).cuda()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if not args.single_level:
            try:
                input_search, target_search = next(valid_queue_iter)
            except:
                valid_queue_iter = iter(valid_queue)
                input_search, target_search = next(valid_queue_iter)
            input_search = input_search.cuda()
            target_search = target_search.cuda(non_blocking=True)

            arch_optimizer.zero_grad()

            logits = model(input_search)
            loss = criterion(logits, target_search)
            model.weights.retain_grad()
            Loss.update(loss.data.item(), n)
            loss.backward()

            sum_grad(model, arch_grads_sum, weight_grads_sum, zz_grads_sum)

            arch_optimizer.step()

            model.alphas.grad.zero_()
            model.weights.grad.zero_()

        optimizer.zero_grad()

        logits = model(input)
        loss = criterion(logits, target)
        model.weights.retain_grad()
        loss.backward()
        
        if args.single_level:
            sum_grad(model, arch_grads_sum, weight_grads_sum, zz_grads_sum)

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        model.alphas.grad.zero_()
        model.weights.grad.zero_()
        
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg, arch_grads_sum, weight_grads_sum, zz_grads_sum


def infer(valid_queue, model, criterion, test=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % args.report_freq == 0:
                if not test:
                    logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
                else:
                    logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def sum_grad(model, arch_grads_sum, weight_grads_sum, zz_grads_sum):
    arch_grads_sum += torch.abs(model.alphas.grad)
    weight_grads_sum += torch.abs(model.weights.grad)
    zz_grads_sum += torch.abs(model.weights.grad - torch.sum(model.c.grad, dim=-1, keepdim=True))


def get_nas_bench(api, set):
    accs = []
    for i in range(len(api)):
        accs.append((i, api.get_more_info(i, set, iepoch=None, hp='200', is_random=False)['test-accuracy']))
    rks = sorted(accs, key=lambda item:item[1], reverse=True)
    rks = sorted(list(enumerate(np.array(rks)[:, 0])), key=lambda item:item[1])
    accs = np.array(accs)[:, 1]
    rks = np.array(rks)[:, 0]
    return accs, rks


if __name__ == '__main__':
    main()
    torch.cuda.empty_cache()
