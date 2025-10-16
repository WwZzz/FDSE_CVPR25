import argparse
import os.path

import numpy as np
import yaml
import flgo.experiment.analyzer as fea
import prettytable


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='name of task', type=str, nargs='*', default=[])
    parser.add_argument('--algorithm', help='name of method', type=str, nargs='*', default=['fedavg'])
    parser.add_argument('--model', help='name of method', type=str, default='')
    parser.add_argument('--simulator', help='name of simulator', type=str, default='')
    parser.add_argument('--config', type=str, help='configuration of hypara', nargs='*', default=['./config/show_run.yml'])
    parser.add_argument('--domain', help='name of method', action='store_true', default=False)
    parser.add_argument('--metric', type=str, help='the name of metric', default='accuracy')
    parser.add_argument('--only_average', help='if only show the averaging result', action='store_true', default=False)
    return parser.parse_known_args()

def max_log(x, op={}):
    res = x.log.get(op['x'], None)
    return max(res) if res is not None else -np.inf
def get_option(x, op={}):
    res = x.option.get(op['x'], None)
    return res
def optimal_by_(x, op={}):
    k = x.log.get(op['x'], None)
    v = x.log.get(op['y'], None)
    if k is None or v is None: return None
    return v[np.argmax(k)]
def optimal_gtest_by_gval(x, op={'metric': 'accuracy'}):
    return optimal_by_(x, {'x': f'val_{op["metric"]}', 'y': f'test_{op["metric"]}'})
def optimal_gtest_by_lval(x, op={'metric': 'accuracy'}):
    return optimal_by_(x, {'x': f'local_val_{op["metric"]}', 'y': f'test_{op["metric"]}'})
def optimal_ltest_by_lval(x, op={'metric': 'accuracy'}):
    return optimal_by_(x, {'x': f'local_val_{op["metric"]}', 'y':f'local_test_{op["metric"]}'})
def optimal_mean_ltest_by_lval(x, op={'metric': 'accuracy'}):
    return optimal_by_(x, {'x': f'local_val_{op["metric"]}', 'y': f'mean_local_test_{op["metric"]}'})
def optimal_std_ltest_by_lval(x, op={'metric': 'accuracy'}):
    return optimal_by_(x, {'x': f'local_val_{op["metric"]}', 'y': f'std_local_test_{op["metric"]}'})
def optimal_round_by_lval(x, op={'metric': 'accuracy'}):
    res = x.log.get(f'local_val_{op["metric"]}', None)
    return np.argmax(res) if res is not None else None
def optimal_round_by_gval(x, op={'metric': 'accuracy'}):
    res = x.log.get(f'val_{op["metric"]}', None)
    return np.argmax(res) if res is not None else None
def max_local_val(x, op={'metric': 'accuracy'}):
    return max_log(x, {'x': f'local_val_{op["metric"]}'})
def max_global_val(x, op={'metric': 'accuracy'}):
    return max_log(x, {'x': f'val_{op["metric"]}'})
def max_global_test(x, op={'metric': 'accuracy'}):
    return max_log(x, {'x': f'test_{op["metric"]}'})
def lr(x, op={}):
    return x.option['learning_rate']
def get_column(tb, name):
    idx = tb.tb.field_names.index(name)
    col_values = [r[idx] for r in tb.tb.rows]
    return col_values
def get_final_res(tb, name):
    res = get_column(tb, name)
    if len(res)==0 or res[0] is None: return np.inf, np.inf
    mean_res = np.mean(res)
    std_res = np.std(res)
    # print(f"Mean {name}:{mean_res}")
    # print(f"std {name}:{std_res}")
    return mean_res, std_res
def get_seed(x, op={}):
    return get_option(x, {'x':'seed'})
def respective_ltest_dist_by_lval(x, op={'metric': 'accuracy'}):
    val_dist = x.data[f'local_val_{op["metric"]}_dist']
    test_dist = x.data[f'local_test_{op["metric"]}_dist']
    val_acc_i = [val_dist[k][op['i']] for k in range(len(test_dist))]
    test_acc_i = [test_dist[k][op['i']] for k in range(len(test_dist))]
    idx = np.argmax(val_acc_i)
    res = test_acc_i[idx]
    return res
def optimal_ltest_dist_by_lval(x, op={'metric': 'accuracy'}):
    idx = np.argmax(x.data[f'local_val_{op["metric"]}'])
    test_dist = x.data[f'local_test_{op["metric"]}_dist']
    num_clients = len(test_dist[0])
    res = []
    for i in range(num_clients):
        test_acc_i = [test_dist[k][i] for k in range(len(test_dist))]
        res.append(test_acc_i[idx])
    res.append(np.mean(np.array(res)))
    return res
def get_client_performance(x, op={'metric': 'accuracy'}):
    idx = np.argmax(x.data[f'local_val_{op["metric"]}'])
    test_dist = x.data[f'local_test_{op["metric"]}_dist']
    test_acc_i = [test_dist[k][op['i']] for k in range(len(test_dist))]
    res = test_acc_i[idx]
    return res

if __name__ == '__main__':
    args = read_args()[0]
    tasks = args.task
    if len(args.config)==1: configs = [args.config[0] for _ in range(len(tasks)*len(args.algorithm))]
    else: configs = args.config
    assert len(configs)==len(tasks)*len(args.algorithm)
    config_id = 0
    for task in tasks:
        for algorithm in args.algorithm:
        # task =args.task
            config = configs[config_id]
            if config!='' and os.path.exists(config):
                with open(config, 'r') as inf:
                    option = yaml.load(inf, Loader=yaml.FullLoader)
            else:
                option = {}
            if args.model != '': option['model'] = [args.model]
            if args.simulator != '': option['simulator'] = [args.simulator]
            # config = args.config
            records = fea.load_records(os.path.join('task', task), algorithm, option)
            tb = fea.Table(records)

            if not args.domain:
                tb.add_column(get_seed)
                tb.add_column(max_local_val, {'metric':args.metric})
                tb.add_column(max_global_test, {'metric':args.metric})
                tb.add_column(max_global_val, {'metric':args.metric})
                tb.add_column(lr)
                tb.add_column(optimal_gtest_by_lval, {'metric':args.metric})
                tb.add_column(optimal_gtest_by_gval, {'metric':args.metric})
                tb.add_column(optimal_ltest_by_lval, {'metric':args.metric})
                tb.add_column(optimal_mean_ltest_by_lval, {'metric':args.metric})
                tb.add_column(optimal_round_by_lval, {'metric':args.metric})
                tb.add_column(optimal_round_by_gval, {'metric':args.metric})
                if not args.only_average: tb.print()
                col_names = [optimal_gtest_by_gval.__name__, optimal_gtest_by_lval.__name__, optimal_ltest_by_lval.__name__, optimal_mean_ltest_by_lval.__name__]
                res = prettytable.PrettyTable(['task', 'algorithm', 'num_records', ]+col_names)
                row = [task, algorithm, len(records)]
                for n in col_names:
                    a,b = get_final_res(tb,  '-'.join([n, args.metric]))
                    row.append("{:.2f}±{:.2f}".format(a*100 ,b*100))
                res.add_row(row)
                print(res)
            else:
                num_clients = int(task.split('_')[-1][1:])
                for k in range(num_clients):
                    tb.add_column(get_client_performance, {'i': k, 'name': f"Client-{k}", 'metric':args.metric})
                tb.add_column(optimal_ltest_by_lval, {'name': 'Weighted-Mean', 'metric':args.metric})
                tb.add_column(optimal_mean_ltest_by_lval, {'name': 'Mean', 'metric':args.metric})
                tb.add_column(optimal_std_ltest_by_lval, {'name': 'Std', 'metric':args.metric})
                row = [f"{len(tb.tb.rows)}", 'averaged', ]
                for k in range(num_clients):
                    mk, sk = get_final_res(tb, f'Client-{k}')
                    row.append("{:.2f}±{:.2f}".format(mk * 100, sk * 100))
                mk, sk = get_final_res(tb, 'Weighted-Mean')
                row.append("{:.2f}±{:.2f}".format(mk * 100, sk * 100))
                mk, sk = get_final_res(tb, 'Mean')
                row.append("{:.2f}±{:.2f}".format(mk * 100, sk * 100))
                mk, sk = get_final_res(tb, 'Std')
                row.append("{:.2f}±{:.2f}".format(mk * 100, sk * 100))
                # get_final_res(tb,  optimal_gtest_by_gval.__name__)
                # get_final_res(tb,  optimal_gtest_by_lval.__name__)
                if args.only_average: tb.tb.clear_rows()
                tb.tb.add_row(row)
                tb.tb.title = f"{task}-{algorithm}"
                tb.print()
