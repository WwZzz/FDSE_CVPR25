import flgo
import flgo.algorithm.fedavg as fedavg
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.partition as fbp

task = './my_task'
flgo.gen_task_by_(mnist, fbp.IIDPartitioner(num_clients=100), task)
flgo.init(task, fedavg, {'gpu':0, 'num_rounds':3}).run()