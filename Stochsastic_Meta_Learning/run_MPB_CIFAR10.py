from subprocess import call

call(['python', 'main_Meta_Bayes.py',
      '--data-source', 'CIFAR10',
      '--data-transform', 'Permute_Labels',
      '--n_train_tasks', '5',

      '--model-name',   'BayesDenseNet', # TODO: implement stochastic 'OmConvNet',
      '--n_test_tasks', '10',
      '--n_meta_train_epochs', '150',
      '--mode', 'MetaTrain',  # 'MetaTrain'  \ 'LoadMetaModel'
      # '--override_eps_std', '1e-3',
      ])


