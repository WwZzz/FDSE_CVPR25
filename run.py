import argparse
import warnings
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import flgo
import flgo.experiment.device_scheduler as ds
import torch.multiprocessing
import yaml
import importlib
import logger
import simulator
import flgo.simulator
import torch

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='name of task', type=str,  nargs='*', default=[])
    parser.add_argument('--algorithm', help='name of method', type=str, nargs='*', default='fedavg')
    parser.add_argument('--model', help = 'model name', type=str, default='')
    parser.add_argument('--simulator', help='test parallel', type=str, default='')
    parser.add_argument('--gpu', nargs='*', help='GPU IDs and empty input is equal to using CPU', type=int, default=[0])
    # parser.add_argument('--seeds', nargs='+', help='seeds', type=int, default=[2,4388,15,333,967])
    parser.add_argument('--seeds', nargs='+', help='seeds', type=int, default=[2,15,333])
    parser.add_argument('--config', type=str, help='configuration of hypara', default='', nargs='*')
    parser.add_argument('--check_interval', help='interval (s) to save checkpoints', type=int, default=-1)
    parser.add_argument('--put_interval', help='interval (s) to put command into devices', type=int, default=10)
    parser.add_argument('--max_pdev', help='interval (s) to put command into devices', type=int, default=7)
    parser.add_argument('--available_interval', help='check availability of devices every x seconds', type=int, default=10)
    parser.add_argument('--mmap', help='mmap',  action="store_true", default=False)
    parser.add_argument('--load_mode', help = 'load_mode', type=str, default='')
    parser.add_argument('--seq', help='run sequencially',  action="store_true", default=False)
    parser.add_argument('--train_parallel', help = 'number of parallel processing', type=int, default=0)
    parser.add_argument('--test_parallel', help='test parallel',  action="store_true", default=False)
    parser.add_argument('--logger', help='test parallel', type=str, default='FullLogger')
    parser.add_argument('--data_root', help = 'the root of dataset', type=str, default='')
    parser.add_argument('--use_cache', help='whether to use cache',  action="store_true", default=False)
    parser.add_argument('--torch_num_threads', help='parameter of torch.set_num_threads', type=int, default=2)
    parser.add_argument('--save_model', help='whether to save the optimal model',  action="store_true", default=False)
    return parser.parse_known_args()

args = read_args()[0]
task = args.task
seeds = args.seeds
gpus = args.gpu
assert len(args.config)>0
if args.data_root!='':
    if os.path.exists(args.data_root) and os.path.isdir(args.data_root):
        flgo.set_data_root(args.data_root)
    else:
        warnings.warn("Failed to change data root.")
optimal_options = []
for config in args.config:
    with open(config, 'r') as inf:
        option = yaml.load(inf, Loader=yaml.FullLoader)
    option['load_mode'] = args.load_mode
    if 'early_stop' in option.keys(): option.pop('early_stop')
    optimal_options.append(option)

def fedrun(tasks, algos=[], optimal_options=[], seeds=[0], Logger=None, model=None, Simulator=flgo.simulator.DefaultSimulator, put_interval=10, available_interval=10, max_processes_per_device=10, mmap=False, seq=False, check_interval=-1):
    assert len(tasks)*len(algos)==len(optimal_options)
    runner_dict = []
    asc = ds.AutoScheduler(args.gpu, put_interval=put_interval, available_interval=available_interval, max_processes_per_device=max_processes_per_device)
    if not seq:
        oid = 0
        for task in tasks:
            for algo in algos:
                optimal_option = optimal_options[oid]
                for seed in seeds:
                    opi = optimal_option.copy()
                    opi.update({'seed': seed, 'use_cache': args.use_cache, 'dataseed':seed })
                    if check_interval > 0: opi.update({'load_checkpoint': algo.__name__, 'save_checkpoint': algo.__name__, 'check_interval': check_interval})
                    runner_dict.append({'task': task, 'algorithm': algo, 'option': opi, 'model':model, 'Logger':Logger, 'Simulator':Simulator})
                oid += 1
        res = flgo.multi_init_and_run(runner_dict, scheduler=asc, mmap=mmap)
    else:
        res = []
        oid = 0
        for task in tasks:
            for algo in algos:
                options = []
                optimal_option = optimal_options[oid]
                for seed in seeds:
                    opi = optimal_option.copy()
                    opi.update({'seed': seed, 'no_tqdm': True, 'dataseed':seed})
                    options.append(opi)
                tmp = flgo.run_in_sequencial(task, algo, options, model, Logger=Logger, mmap=mmap)
                oid += 1
                res.append(tmp)
    return res

if __name__=='__main__':
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    if args.torch_num_threads>0: torch.set_num_threads(args.torch_num_threads)
    algos = []
    acce = False
    for algoname in args.algorithm:
        modules = [".".join(["algorithm", algoname]), ".".join(["develop", algoname]),
                   ".".join(["flgo", "algorithm", algoname])]
        algo = None
        if args.train_parallel > 0:
            try:
                algo = importlib.import_module(".".join(["algorithm", "accelerate", algoname]))
                acce = True
            except:
                algo = None
                warnings.warn(f"There is no acceleration support for {algoname}")
        if algo is None:
            for m in modules:
                try:
                    algo = importlib.import_module(m)
                    break
                except ModuleNotFoundError:
                    continue
        if algo is None: raise ModuleNotFoundError("{} was not found".format(algo))
        algos.append(algo)
    model  = None
    if args.model != '':
        try:
            model = getattr(algo, args.model)
        except:
            model = None
        if model is None:
            try:
                model = importlib.import_module(args.model)
            except:
                print("using default model")
                model = None
    if acce and args.train_parallel>0:
        for optimal_option in optimal_options:
            optimal_option['num_parallels'] =args.train_parallel
            optimal_option['parallel_type'] = 'else'
    if args.test_parallel:
        for optimal_option in optimal_options:
            optimal_option['test_parallel'] = True
    if args.save_model:
        for optimal_option in optimal_options:
            optimal_option['save_optimal'] = True
    Logger = getattr(logger, args.logger)
    Simulator = getattr(simulator, args.simulator) if args.simulator!='' else flgo.simulator.DefaultSimulator
    tasks = [os.path.join('task', task) for task in args.task]
    fedrun(tasks, algos, optimal_options=optimal_options, seeds=seeds, Logger=Logger, model=model, Simulator=Simulator, put_interval=args.put_interval, available_interval=args.available_interval, max_processes_per_device=args.max_pdev, mmap=args.mmap, seq=args.seq, check_interval=args.check_interval)