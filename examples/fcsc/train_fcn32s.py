#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp

os.environ['MPLBACKEND'] = 'Agg'  # NOQA

import chainer
import fcn

import imp
train_fcn32s_parent = imp.load_source('train_fcn32s_parent', '../voc/train_fcn32s.py')  # NOQA
get_trainer = train_fcn32s_parent.get_trainer  # NOQA

from dataset import FCSCDataset


here = osp.dirname(osp.abspath(__file__))


def transform_to_augment(in_data):
    import mvtk
    img, lbl = in_data
    obj_data = mvtk.aug.augment_object_data(
        [{'img': img, 'lbl': lbl}], fit_output=True)
    obj_datum = next(obj_data)
    img, lbl = obj_datum['img'], obj_datum['lbl']
    return img, lbl


def get_data():
    dataset_train = FCSCDataset(split='train')

    class_names = dataset_train.class_names

    dataset_train = chainer.datasets.TransformDataset(
        dataset_train, transform_to_augment)
    dataset_train = chainer.datasets.TransformDataset(
        dataset_train, fcn.datasets.transform_lsvrc2012_vgg16)
    iter_train = chainer.iterators.SerialIterator(
        dataset_train, batch_size=1)

    dataset_valid = FCSCDataset(split='val')
    iter_valid_raw = chainer.iterators.SerialIterator(
        dataset_valid, batch_size=1, repeat=False, shuffle=False)
    dataset_valid = chainer.datasets.TransformDataset(
        dataset_valid, transform_to_augment)
    dataset_valid = chainer.datasets.TransformDataset(
        dataset_valid, fcn.datasets.transform_lsvrc2012_vgg16)
    iter_valid = chainer.iterators.SerialIterator(
        dataset_valid, batch_size=1, repeat=False, shuffle=False)

    return class_names, iter_train, iter_valid, iter_valid_raw


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    args = parser.parse_args()

    args.model = 'FCN32s'
    args.lr = 1e-5
    args.momentum = 0.99
    args.weight_decay = 0.0005

    args.max_iteration = 1000
    args.interval_print = 20
    args.interval_eval = 100

    now = datetime.datetime.now()
    args.timestamp = now.isoformat()
    args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S'))

    # data
    class_names, iter_train, iter_valid, iter_valid_raw = get_data()
    n_class = len(class_names)

    # model
    vgg = fcn.models.VGG16()
    chainer.serializers.load_npz(vgg.download(), vgg)
    model = fcn.models.FCN32s(n_class=n_class)
    model.init_from_vgg16(vgg)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # optimizer
    optimizer = chainer.optimizers.Adam(alpha=args.lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=args.weight_decay))
    model.upscore.disable_update()

    # trainer
    trainer = get_trainer(optimizer, iter_train, iter_valid, iter_valid_raw,
                          class_names, args)
    trainer.run()


if __name__ == '__main__':
    main()
