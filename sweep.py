import wandb
import argparse
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('--wandb_proj', type=str, default='samediff',
                    help='Name of WandB project to store the run in.')
parser.add_argument('--wandb_entity', type=str, default='samediff', help='Team to send run to.')

parser.add_argument('-m', '--model_type', help='Model to train: resnet, vit, clip_rn, clip_vit.',
                    type=str, required=True)
parser.add_argument('--patch_size', type=int, default=32, help='Size of patch (eg. 16 or 32).')
parser.add_argument('--cnn_size', type=int, default=50,
                    help='Number of layers for ResNet (eg. 50 = ResNet-50).', required=False)
parser.add_argument('--feature_extract', action='store_true', default=False,
                    help='Only train the final layer; freeze all other layers.')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Use ImageNet pretrained models. If false, models are trained from scratch.')
parser.add_argument('--num_gpus', type=int, default=1, required=False)

args = parser.parse_args()

# Model formatted strings for sweep name
model_names = {
    'resnet': f'ResNet-{args.cnn_size}',
    'vit': f'ViT-B/{args.patch_size}',
    'clip_rn': f'ResNet-{args.cnn_size}',
    'clip_vit': f'ViT-B/{args.patch_size}'
    }
sweep_name = model_names[args.model_type]

# Define command structure + specify sweep name
commands = ['${env}', '${interpreter}', '${program}', '--unaligned', '--rotation', '--scaling']

if args.pretrained:
    commands += ['--pretrained']
    if 'clip' in args.model_type:
        sweep_name = f'CLIP {sweep_name}'
    else:
        sweep_name = f'ImageNet {sweep_name}'
else:
    sweep_name = f'From Scratch {sweep_name}'
    
if args.feature_extract:
    commands += '--feature_extract'
    sweep_name += ' Feature Extract'
    
commands += ['${args}']

sweep_configuration = {
    'method': 'grid',
    'program': 'train.py',
    'command': commands,
    'name': sweep_name,
    'parameters': {
        'train_datasets': {'values': ['OBJECTSALL',
                                      'GRAYSCALE_OBJECTSALL',
                                      'MASK_OBJECTSALL',
                                      'DEVELOPMENTAL',
                                      'GRAYSCALE_DEVELOPMENTAL',
                                      'MASK_DEVELOPMENTAL',
                                      'ALPHANUMERIC',
                                      'SQUIGGLES']},
        'lr': {'values': [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]},
        'lr_scheduler': {'values': ['reduce_on_plateau', 'exponential']},
        'n_train_tokens': {'values': [1200]},
        'n_val_tokens': {'values': [300]},
        'n_test_tokens': {'values': [100]},
        'patch_size': {'values': [args.patch_size]},
        'num_epochs': {'values': [70]},
        'wandb_proj': {'values': [args.wandb_proj]},
        'wandb_entity': {'values': [args.wandb_entity]},
        'wandb_cache_dir': {'values': ['../../../home/art481/.cache']},
        'num_gpus': {'values': [args.num_gpus]},
        'model_type': {'values': [args.model_type]},
        'batch_size': {'values': [128]}
        }
    }

if args.model_type == 'resnet' or args.model_type == 'clip_rn':
    sweep_configuration['parameters'].pop('patch_size')
    sweep_configuration['parameters']['cnn_size'] = {'values': [args.cnn_size]}
    
sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.wandb_proj, entity=args.wandb_entity)
<<<<<<< HEAD

if args.slurm:  # Parallelize across multiple GPUs
    sp = []
    for i in range(args.n_gpus):
        sp.append(subprocess.Popen(['srun',
                    'agents.sh',
                    str(i),
                    f'{args.wandb_entity}/{args.wandb_proj}/{sweep_id}']))
    exit_codes = [p.wait() for p in sp]
else:
    wandb.agent(sweep_id=sweep_id, project=args.wandb_proj, entity=args.wandb_entity)
=======
wandb.agent(sweep_id=sweep_id, project=args.wandb_proj, entity=args.wandb_entity)
>>>>>>> 47da5c4faee0a33857e9216faab830fe7a9ed6f6
