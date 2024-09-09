import os
from datas.benchmark import Benchmark
from datas.us1k import US1K
from torch.utils.data import DataLoader


def create_datasets(args):
    if args.training_dataset == 'us1k':
        us1k = US1K(
            os.path.join(args.data_path, 'US1K/US1K_train_HR'), 
            os.path.join(args.data_path, 'US1K/US1K_train_LR_bicubic'), 
            os.path.join(args.data_path, 'us1k_cache'),
            train=True, 
            augment=args.data_augment, 
            scale=args.scale, 
            colors=args.colors, 
            patch_size=args.patch_size, 
            repeat=args.data_repeat, 
            add_noise=args.data_add_noise,
            cutout=args.cutout,
        )
        train_dataloader = DataLoader(dataset=us1k, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False)
    else:
        raise NotImplementedError("=== dataset [{}] is not found ===".format(args.training_dataset))
   
    valid_dataloaders = []
    if 'CCA-US' in args.eval_sets:
        ui5_hr_path = os.path.join(args.data_path, 'benchmark/UI5/HR')
        ui5_lr_path = os.path.join(args.data_path, 'benchmark/UI5/LR_bicubic')
        ui5  = Benchmark(ui5_hr_path, ui5_lr_path, scale=args.scale, colors=args.colors)
        valid_dataloaders += [{'name': 'CCA-US', 'dataloader': DataLoader(dataset=ui5, batch_size=1, shuffle=True)}]

    if 'US-CASE' in args.eval_sets:
        us15_hr_path = os.path.join(args.data_path, 'benchmark/US15/HR')
        us15_lr_path = os.path.join(args.data_path, 'benchmark/US15/LR_bicubic')
        us15  = Benchmark(us15_hr_path, us15_lr_path, scale=args.scale, colors=args.colors)
        valid_dataloaders += [{'name': 'US-CASE', 'dataloader': DataLoader(dataset=us15, batch_size=1, shuffle=True)}]

    if 'US1K_23' in args.eval_sets:
        us1k_23_hr_path = os.path.join(args.data_path, 'benchmark/US1K_23/HR')
        us1k_23_lr_path = os.path.join(args.data_path, 'benchmark/US1K_23/LR_bicubic')
        us1k_23  = Benchmark(us1k_23_hr_path, us1k_23_lr_path, scale=args.scale, colors=args.colors)
        valid_dataloaders += [{'name': 'US1K_23', 'dataloader': DataLoader(dataset=us1k_23, batch_size=1, shuffle=True)}]
    
    if len(valid_dataloaders) == 0:
        print('select no dataset for evaluation!')
    else:
        selected = ''
        for i in range(0, len(valid_dataloaders)):
            selected += " " + valid_dataloaders[i]['name']
        print('##=== select {} for evaluation! ===##'.format(selected))

    return train_dataloader, valid_dataloaders