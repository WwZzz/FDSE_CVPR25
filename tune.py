import sys
import os.path
import warnings
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import importlib
import argparse
import os.path
import flgo
import torch.multiprocessing as mlp
import yaml
import logger
import simulator
import flgo.simulator

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', help='algorithm name', type=str, default=['fedavg'], nargs='*')
    parser.add_argument('--task', help='task name', type=str, default=['cifar10_iid_c100'], nargs='*')
    parser.add_argument('--gpu', nargs='*', help='GPU IDs and empty input is equal to using CPU', type=int, default=[0])
    parser.add_argument('--config', help='congiguration', type=str, default=[], nargs='*')
    parser.add_argument('--model', help = 'model name', type=str, default=[], nargs='*')
    parser.add_argument('--simulator', help='test parallel', type=str, default='')
    parser.add_argument('--check_interval', help='interval (s) to save checkpoints', type=int, default=-1)
    parser.add_argument('--put_interval', help='interval (s) to put command into devices', type=int, default=5)
    parser.add_argument('--max_pdev', help='interval (s) to put command into devices', type=int, default=7)
    parser.add_argument('--available_interval', help='check availability of devices every x seconds', type=int, default=5)
    parser.add_argument('--memory', help='mean memory occupation', type=float, default=1000)
    parser.add_argument('--no_dynmem', help='no_dynmem',  action="store_true", default=False)
    parser.add_argument('--mmap', help='mmap',  action="store_true", default=False)
    parser.add_argument('--load_mode', help = 'load_mode', type=str, default='')
    parser.add_argument('--seq', help='tune sequencially',  action="store_true", default=False)
    parser.add_argument('--train_parallel', help='number of parallel processing',   type=int, default=0)
    parser.add_argument('--test_parallel', help='test parallel',  action="store_true", default=False)
    parser.add_argument('--logger', help='test parallel', type=str, default=['TuneLogger'], nargs='*')
    parser.add_argument('--data_root', help = 'the root of dataset', type=str, default='')
    parser.add_argument('--use_cache', help='whether to use cache',  action="store_true", default=False)
    return parser.parse_known_args()

if __name__=='__main__':
    mlp.set_sharing_strategy("file_system")
    mlp.set_start_method("spawn", force=True)
    args = read_args()[0]
    if args.data_root != '':
        if os.path.exists(args.data_root) and os.path.isdir(args.data_root):
            flgo.set_data_root(args.data_root)
        else:
            warnings.warn("Failed to change data root.")
    tasks = args.task
    Simulator = getattr(simulator, args.simulator) if args.simulator!='' else flgo.simulator.DefaultSimulator
    if len(tasks)==1:
        task = tasks[0]
        assert len(args.logger)==1
        config_files = args.config
        configs = []
        if len(config_files)>0:
            assert len(args.algorithm)==len(config_files)
            for config_i in config_files:
                with open(config_i, 'r') as inf:
                    config_data_i = yaml.load(inf, yaml.Loader)
                configs.append(config_data_i)
        else:
            config_files = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task', task, 'config.yml')
            if os.path.exists(config_files):
                with open(config_files, 'r') as inf:
                    config = yaml.load(inf, yaml.Loader)
            else:
                config = {}
            algo_para_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'algo_para.yml')
            with open(algo_para_config, 'r') as inf:
                algo_para_config = yaml.load(inf, yaml.Loader)
            for algo in args.algorithm:
                config_tmp = config.copy()
                algo_para = algo_para_config.get(algo, None)
                if algo_para is not None: config_tmp['algo_para'] = algo_para
                configs.append(config_tmp)
        for config in configs:
            config['load_mode'] = args.load_mode
            config['use_cache'] = args.use_cache
        import flgo.experiment.device_scheduler as fed
        scheduler = None if args.gpu is None else fed.AutoScheduler(args.gpu, put_interval=args.put_interval, available_interval=args.available_interval, mean_memory_occupated=args.memory, dynamic_memory_occupated=not args.no_dynmem, max_processes_per_device=args.max_pdev)
        #
        # paras = config
        # paras['load_mode'] = args.load_mode
        algos = []
        for algo_id, algo_name in enumerate(args.algorithm):
            algo = None
            acce = False
            modules = [".".join(["algorithm", algo_name]), ".".join(["develop", algo_name]),".".join(["flgo", "algorithm", algo_name])]
            if args.train_parallel>0:
                try:
                    algo = importlib.import_module(".".join(["algorithm", "accelerate", algo_name]))
                    acce = True
                except:
                    algo = None
                    warnings.warn(f"There is no acceleration support for {algo_name}")
            if algo is None:
                for m in modules:
                    try:
                        algo = importlib.import_module(m)
                        break
                    except ModuleNotFoundError:
                        continue
            if algo is None: raise ModuleNotFoundError("{} was not found".format(algo))
            algos.append(algo)
            if acce and args.train_parallel>0:
                configs[algo_id]['num_parallels'] = args.train_parallel
                configs[algo_id]['parallel_type'] = 'else'
        task = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task', task)
        models = []
        if len(args.model) > 0:
            assert len(args.model)==len(args.algorithm) or len(args.model)==1
            if len(args.model)==len(args.algorithm) and len(args.model)>1:
                for algo, amodel in zip(algos, args.model):
                    model = None
                    try:
                        model = getattr(algo, amodel)
                    except:
                        model = None
                    if model is None:
                        try:
                            model = importlib.import_module(amodel)
                        except:
                            print("using default model")
                            model = None
                    models.append(model)
            else:
                amodel = args.model[0]
                for algo in algos:
                    model = None
                    try:
                        model = getattr(algo, amodel)
                    except:
                        model = None
                    if model is None:
                        try:
                            model = importlib.import_module(amodel)
                        except:
                            print("using default model")
                            model = None
                    models.append(model)
        if args.test_parallel:
            for config in configs:
                config['test_parallel'] = True
        Logger = getattr(logger, args.logger[0])
        if args.check_interval >0:
            for algo, config in zip(algos, configs):
                config['check_interval'] = args.check_interval
                config['save_checkpoint'] = algo.__name__
                config['load_checkpoint'] = algo.__name__
        if args.seq:
            if len(models)==0: models = [None for _ in algos]
            for algo, para, model in zip(algos, configs, models):
                para['gpu'] = args.gpu
                para['no_tqdm'] = True
                res = flgo.tune_sequencially(task, algo, para, model=model, Logger=Logger, Simulator=Simulator, mmap=args.mmap, target_path=os.path.join(os.path.dirname(__file__), 'config'))
        else:
            if len(algos)==1:
                algo = algos[0]
                paras = configs[0]
                model = models[0] if len(models)>0 else None
                paras['no_tqdm'] = True
                res = flgo.tune(task, algo, paras, model=model, Logger=Logger, Simulator=Simulator, scheduler=scheduler, mmap=args.mmap, target_path=os.path.join(os.path.dirname(__file__), 'config'))
            else:
                for config in configs: config['no_tqdm'] = True
                task_dict = {'task':task, 'algorithm':algos, 'option': configs, 'Logger':Logger, 'Simulator':Simulator, 'model':models if len(models)>0 else None}
                flgo.multi_tune(task_dict, scheduler=scheduler, target_path=os.path.join(os.path.dirname(__file__), 'config'))
    else:
        import flgo.experiment.device_scheduler as fed
        scheduler = None if args.gpu is None else fed.AutoScheduler(args.gpu, put_interval=args.put_interval,
                                                                    available_interval=args.available_interval,
                                                                    mean_memory_occupated=args.memory,
                                                                    dynamic_memory_occupated=not args.no_dynmem,
                                                                    max_processes_per_device=args.max_pdev)
        algo_para_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'algo_para.yml')
        with open(algo_para_config, 'r') as inf:
            algo_para_config = yaml.load(inf, yaml.Loader)
        assert len(args.model)==0 or len(args.model)==len(tasks)
        assert len(args.logger)==len(tasks) or len(args.logger)==1
        acces = [False for _ in args.algorithm]
        algos = []
        for algo_id, algo_name in enumerate(args.algorithm):
            algo = None
            modules = [".".join(["algorithm", algo_name]), ".".join(["develop", algo_name]),
                       ".".join(["flgo", "algorithm", algo_name])]
            if args.train_parallel > 0:
                try:
                    algo = importlib.import_module(".".join(["algorithm", "accelerate", algo_name]))
                    acces[algo_id] = True
                except:
                    algo = None
                    warnings.warn(f"There is no acceleration support for {algo_name}")
            if algo is None:
                for m in modules:
                    try:
                        algo = importlib.import_module(m)
                        break
                    except ModuleNotFoundError:
                        continue
            if algo is None: raise ModuleNotFoundError("{} was not found".format(algo))
            algos.append(algo)
        task_dicts = []
        for task_id, task in enumerate(tasks):
            config_files = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task', task, 'config.yml')
            configs = []
            if os.path.exists(config_files):
                with open(config_files, 'r') as inf:
                    config = yaml.load(inf, yaml.Loader)
            else:
                warnings.warn(f"There exists no default configuration for task {task}")
                config = {}
            for algo_name in args.algorithm:
                config_tmp = config.copy()
                algo_para = algo_para_config.get(algo_name, None)
                if algo_para is not None: config_tmp['algo_para'] = algo_para
                configs.append(config_tmp)
            for config in configs:
                config['load_mode'] = args.load_mode
                config['use_cache'] = args.use_cache
            for algo_id, algo in enumerate(args.algorithm):
                if acces[algo_id] and args.train_parallel > 0:
                    configs[algo_id]['num_parallels'] = args.train_parallel
                    configs[algo_id]['parallel_type'] = 'else'
            task = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task', task)
            models = []
            if len(args.model)>0:
                model_name = args.model[task_id]
                for algo in algos:
                    model = None
                    try:
                        model = getattr(algo, model_name)
                    except:
                        model = None
                    if model is None:
                        try:
                            model = importlib.import_module(model_name)
                        except:
                            print("using default model")
                            model = None
                    models.append(model)
            if args.test_parallel:
                for config in configs:
                    config['test_parallel'] = True
            if len(args.logger)==1:
                Logger = getattr(logger, args.logger[0])
            else:
                Logger = getattr(logger, args.logger[task_id])
            if args.check_interval>0:
                for algo, config in zip(algos, configs):
                    config['check_interval'] = args.check_interval
                    config['save_checkpoint'] = algo.__name__
                    config['load_checkpoint'] = algo.__name__
            for config in configs:
                config['no_tqdm'] = True
            task_dict = {'task': task, 'algorithm': algos, 'option': configs, 'Logger': Logger, 'Simulator':Simulator, 'model': models if len(models) > 0 else None}
            task_dicts.append(task_dict)
        if args.seq:
            for task_dict in task_dicts:
                task = task_dict['task']
                algos = task_dict['algorithm']
                models = task_dict['model'] if task_dict['model'] is not None else [None for _ in range(len(algos))]
                options = task_dict['option']
                Logger = task_dict['Logger']
                Simulator = task_dict['Simulator']
                for algo, para, model in zip(algos, options, models):
                    para['gpu'] = args.gpu
                    res = flgo.tune_sequencially(task, algo, para, model=model, Logger=Logger, Simulator=Simulator, mmap=args.mmap,
                                                 target_path=os.path.join(os.path.dirname(__file__), 'config'))
        else:
            flgo.multi_tune(task_dicts, scheduler=scheduler, target_path=os.path.join(os.path.dirname(__file__), 'config'))

