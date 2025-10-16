import flgo
import flgo.benchmark.cifar10_classification as cifar10
import flgo.benchmark.partition as fbp
# task = 'task/cifar10_dir1.0_c100'
# ylim = [0., 1]
#
# # task = 'task/mnist_dir1.0_c100'
# # ylim = [0.97, 0.99]
# # flgo.gen_task_by_(cifar10, fbp.DirichletPartitioner(num_clients=100, alpha=1.0), task)
# import flgo.experiment.analyzer as fea
#
# records = fea.load_records(task, ['fedavg', 'divtrain'])
# painter = fea.Painter(records)
# painter.create_figure(fea.Curve, {
#     'args':{'x':'communication_round', 'y':'local_test_accuracy'},\
#     'fig_option':{'ylim':ylim, 'title':'test_accuracy'}
# })
#
# painter.create_figure(fea.Curve, {
#     'args':{'x':'communication_round', 'y':'local_val_accuracy'},
#     'fig_option':{'ylim':ylim, 'title':'val_accuracy'}
# })

task = 'task/cifar10_iid_c100'
flgo.gen_task_by_(cifar10, fbp.IIDPartitioner(num_clients=100), task)