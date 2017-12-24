from subprocess import call
import os
import timeit, time

alpha = 0.01
n_meta_train_grad_steps = 2

run_name = 'Labels_Alpha_{}_Grads_{}'.format(alpha, n_meta_train_grad_steps)
run_name = run_name.replace('.','_')

log_dir = 'Logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

start_time = timeit.default_timer()

for n_meta_test_grad_steps in range(1, 21):
    if n_meta_test_grad_steps == 1:
        mode = 'MetaTrain'
    else:
        mode = 'LoadMetaModel'

    call(['python', 'main_MAML.py',
          '--data-source', 'MNIST',
          '--n_train_tasks', '5',
          '--data-transform', 'Permute_Labels',
          '--model-name', 'ConvNet3',
          # MAML hyper-parameters:
          '--alpha', str(alpha),
          '--n_meta_train_grad_steps', str(n_meta_train_grad_steps),
          '--n_meta_train_iterations', '300', #  '300',
          '--meta_batch_size', '32',
          '--n_meta_test_grad_steps', str(n_meta_test_grad_steps),
          '--n_test_tasks', '100',  #  '100',
          '--limit_train_samples_in_test_tasks', '2000',
          '--log-file', log_dir+'/'+run_name,
          '--meta_model_file_name', run_name,
          '--mode', mode,
          ])

stop_time = timeit.default_timer()
print('Total runtime: ' +
      time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)))