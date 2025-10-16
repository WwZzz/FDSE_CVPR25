import argparse
import os.path
import numpy as np
import yaml
import flgo.experiment.analyzer as fea

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='name of task', nargs='*', type=str, default=[])
    parser.add_argument('--algorithm', help='name of method', nargs='*', type=str, default=['fedavg'])
    parser.add_argument('--model', help='name of model', type=str, default='')
    parser.add_argument('--config', type=str, help='configuration of hypara', default='')
    parser.add_argument('--metric', type=str, help='the name of metric', default='accuracy')
    parser.add_argument('--simulator', help='name of simulator', type=str, default='')
    return parser.parse_known_args()

def get_column(tb, name):
    idx = tb.tb.field_names.index(name)
    col_values = [r[idx] for r in tb.tb.rows]
    return col_values
def max_log(x, op={}):
    res = x.log.get(op['x'], None)
    return max(res) if res is not None else -np.inf

def max_local_val(x, op={'metric': 'accuracy'}):
    return max_log(x, {'x': 'local_val_'+op['metric']})

def max_global_val(x, op={'metric': 'accuracy'}):
    return max_log(x, {'x': 'val_'+op['metric']})

def col_model(x, op={}):
    return x.option['model']

def lr(x, op={}):
    return x.option['learning_rate']

def optimal_round_by_val(x, op={'metric': 'accuracy'}):
    res = x.log.get('val_'+op['metric'], None)
    return np.argmax(res) if res is not None else -np.inf

def optimal_round_by_local_val(x, op={'metric': 'accuracy'}):
    res = x.log.get('local_val_'+op['metric'], None)
    return np.argmax(res) if res is not None else -np.inf

if __name__ == '__main__':
    args = read_args()[0]
    if args.config!='' and os.path.exists(args.config):
        with open(args.config, 'r') as inf:
            option = yaml.load(inf, Loader=yaml.FullLoader)
    else: option = {}
    config = args.config
    if args.model!='': option['model'] = [args.model]
    if args.simulator!='': option['simulator'] = [args.simulator]
    for task in args.task:
        for algorithm in args.algorithm:
            records = fea.load_records(os.path.join('task', task), algorithm, option)
            print(f"Number of Records: {len(records)}")
            painter = fea.Painter(records)
            try:
                painter.create_figure(fea.Curve, {'args':{'x':'communication_round', 'y':f"val_{config['metric']}"}})
            except:
                pass
            try:
                painter.create_figure(fea.Curve, {'args':{'x':'communication_round', 'y':f"local_val_{config['metric']}"}})
            except:
                pass
            tb = fea.Table(records)
            tb.add_column(max_local_val, {'metric':args.metric})
            tb.add_column(max_global_val, {'metric':args.metric})
            tb.add_column(lr)
            tb.add_column(col_model)
            tb.add_column(optimal_round_by_val, {'metric':args.metric})
            tb.add_column(optimal_round_by_local_val, {'metric':args.metric})
            sort_key =  max_local_val.__name__+'-'+args.metric
            gv = get_column(tb, max_global_val.__name__+'-'+args.metric)
            if len(gv)>0 and gv[0] is not None and gv[0]!=-np.inf:
                sort_key = max_global_val.__name__+'-'+args.metric
            tb.tb.sortby = sort_key
            tb.tb.title = f"{task}-{algorithm}"
            tb.print()