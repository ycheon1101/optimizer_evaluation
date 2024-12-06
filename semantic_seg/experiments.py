# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import itertools
import logging
import math


def get_model_base(architecture, backbone):
    return {
        'dlv3': '_base_/models/deeplabv3_r50-d8.py',
    }[architecture]


def get_pretraining_file(backbone):
    return {
        'r50v1c': 'open-mmlab://resnet50_v1c',
    }[backbone]


def get_backbone_cfg(backbone):
    return {
        'r50v1c': {
            'depth': 50
        },
    }[backbone]


# def update_decoder_in_channels(cfg, architecture, backbone):
#     cfg.setdefault('model', {}).setdefault('decode_head', {})
#     if 'sfa' in architecture:
#         cfg['model']['decode_head']['in_channels'] = 512
#     return cfg


def setup_rcs(cfg, temperature):
    cfg.setdefault('data', {}).setdefault('train', {})
    cfg['data']['train']['rare_class_sampling'] = dict(
        min_pixels=3000, class_temp=temperature, min_crop_ratio=0.5)
    return cfg


def generate_experiment_cfgs(id):

    def config_from_vars():
        cfg = {'_base_': ['_base_/default_runtime.py'], 'n_gpus': n_gpus}
        if seed is not None:
            cfg['seed'] = seed

        # Setup model config
        architecture_mod = architecture
        model_base = get_model_base(architecture_mod, backbone)
        cfg['_base_'].append(model_base)
        cfg['model'] = {
            'pretrained': get_pretraining_file(backbone),
            'backbone': get_backbone_cfg(backbone),
        }

        # Setup UDA config
        if uda == 'target-only':
            cfg['_base_'].append(f'_base_/datasets/{target}_half_{crop}.py')
        elif uda == 'source-only':
            cfg['_base_'].append(
                f'_base_/datasets/{source}_to_{target}_{crop}.py')

        cfg['data'] = dict(
            samples_per_gpu=batch_size,
            workers_per_gpu=workers_per_gpu,
            train={})

        cfg['_base_'].extend(
            [f'_base_/schedules/{opt}.py', f'_base_/schedules/{schedule}.py'])
        cfg['optimizer'] = {'lr': lr}
        cfg['optimizer'].setdefault('paramwise_cfg', {})
        cfg['optimizer']['paramwise_cfg'].setdefault('custom_keys', {})
        opt_param_cfg = cfg['optimizer']['paramwise_cfg']['custom_keys']
        if pmult:
            opt_param_cfg['head'] = dict(lr_mult=10.)

        # Setup runner
        cfg['runner'] = dict(type='IterBasedRunner', max_iters=iters)
        cfg['checkpoint_config'] = dict(
            by_epoch=False, interval=iters, max_keep_ckpts=1)
        cfg['evaluation'] = dict(interval=iters // 10, metric='mIoU')

        # Construct config name
        uda_mod = uda
        cfg['name'] = f'{source}2{target}_{uda_mod}_{architecture_mod}_' \
                      f'{backbone}_{schedule}'
        cfg['exp'] = id
        cfg['name_dataset'] = f'{source}2{target}'
        cfg['name_architecture'] = f'{architecture_mod}_{backbone}'
        cfg['name_encoder'] = backbone
        cfg['name_decoder'] = architecture_mod
        cfg['name_uda'] = uda_mod
        cfg['name_opt'] = f'{opt}_{lr}_pm{pmult}_{schedule}' \
                          f'_{n_gpus}x{batch_size}_{iters // 1000}k'
        if seed is not None:
            cfg['name'] += f'_s{seed}'
        cfg['name'] = cfg['name'].replace('.', '').replace('True', 'T') \
            .replace('False', 'F').replace('cityscapes', 'cs') \
            .replace('synthia', 'syn') \
            .replace('darkzurich', 'dzur')
        return cfg

    # -------------------------------------------------------------------------
    # Set some defaults
    # -------------------------------------------------------------------------
    cfgs = []
    n_gpus = 1
    batch_size = 2
    iters = 60000
    opt, lr, schedule, pmult = 'adamw', 0.00006, 'poly10warm', True
    crop = '512x512'
    datasets = [
        ('gta', 'cityscapes'),
    ]
    architecture = None
    workers_per_gpu = 0
    # -------------------------------------------------------------------------
    # Experiment
    # -------------------------------------------------------------------------
    # Experiment 1: original version
    if id == 100:
       seeds = [1234]
       experiment_types = ['original']
       models = [
           ('dlv3', 'r50v1c'),
       ]
       udas = ['target-only']
       opts = [
           ('rmsprop', 1e-4, 'original', True),
           ('sgd', 2.5e-4, 'original', True),
           ('adamw', 0.00006, 'original', True),
           ('adagrad', 1e-3, 'original', True),
           ('adam', 0.00006, 'original', True),


       ]
       for (source, target), (architecture, backbone), \
           (opt, lr, schedule, pmult), uda, seed, exp_type in \
               itertools.product(datasets, models, opts, udas, seeds, experiment_types):
           cfg = config_from_vars()
           cfgs.append(cfg)

    # Experiment 2: with poly10 scheduler
    elif id == 101:
       seeds = [1234]
       experiment_types = ['schedules-poly10']
       models = [
           ('dlv3', 'r50v1c'),
       ]
       udas = ['target-only']
       opts = [
           ('rmsprop', 1e-4, 'poly10', True),
           ('sgd', 2.5e-4, 'poly10', True),
           ('adamw', 0.00006, 'poly10', True),
           ('adagrad', 1e-3, 'poly10', True),
           ('adam', 0.00006, 'poly10', True),


       ]
       for (source, target), (architecture, backbone), \
           (opt, lr, schedule, pmult), uda, seed, exp_type in \
               itertools.product(datasets, models, opts, udas, seeds, experiment_types):
           cfg = config_from_vars()
           cfgs.append(cfg)

    # Experiment 3: with polylr scheduler
    elif id == 102:
       seeds = [1234]
       experiment_types = ['schedules-polylr']
       models = [
           ('dlv3', 'r50v1c'),
       ]
       udas = ['target-only']
       opts = [
           ('rmsprop', 1e-4, 'polylr', True),
           ('sgd', 2.5e-4, 'polylr', True),
           ('adamw', 0.00006, 'polylr', True),
           ('adagrad', 1e-3, 'polylr', True),
           ('adam', 0.00006, 'polylr', True),


       ]
       for (source, target), (architecture, backbone), \
           (opt, lr, schedule, pmult), uda, seed, exp_type in \
               itertools.product(datasets, models, opts, udas, seeds, experiment_types):
           cfg = config_from_vars()
           cfgs.append(cfg)

    # Experiment 4: with warmup
    elif id == 103:
       seeds = [1234]
       experiment_types = ['schedules-warmup']
       models = [
           ('dlv3', 'r50v1c'),
       ]
       udas = ['target-only']
       opts = [
           ('rmsprop', 1e-4, 'poly10warm', True),
           ('sgd', 2.5e-4, 'poly10warm', True),
           ('adamw', 0.00006, 'poly10warm', True),
           ('adagrad', 1e-3, 'poly10warm', True),
           ('adam', 0.00006, 'poly10warm', True),


       ]
       for (source, target), (architecture, backbone), \
           (opt, lr, schedule, pmult), uda, seed, exp_type in \
               itertools.product(datasets, models, opts, udas, seeds, experiment_types):
           cfg = config_from_vars()
           cfgs.append(cfg)

    # Experiment 5: GTA5 -> Cityscapes with warmup
    elif id == 104:
       seeds = [1234]
       experiment_types = ['uda-schedules-warmup']
       models = [
           ('dlv3', 'r50v1c'),
       ]
       udas = ['source-only']
       opts = [
           ('rmsprop', 1e-4, 'poly10warm', True),
           ('sgd', 2.5e-4, 'poly10warm', True),
           ('adamw', 0.00006, 'poly10warm', True),
           ('adagrad', 1e-3, 'poly10warm', True),
           ('adam', 0.00006, 'poly10warm', True),


       ]
       for (source, target), (architecture, backbone), \
           (opt, lr, schedule, pmult), uda, seed, exp_type in \
               itertools.product(datasets, models, opts, udas, seeds, experiment_types):
           cfg = config_from_vars()
           cfgs.append(cfg)
  
    
    else:
        raise NotImplementedError('Unknown id {}'.format(id))

    return cfgs
