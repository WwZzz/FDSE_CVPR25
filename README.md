# FDSE_CVPR25
(CVPR2025) The implementation of the paper 'Federated Learning with Domain Shift Eraser'. 
<img width="1032" height="734" alt="image" src="https://github.com/user-attachments/assets/11174369-5233-4731-a8ef-5eacda8f684e" />

# Installation

1. Install pytorch
2. `pip install flgo`

# Running Command
```shell
python run_single.py --task office_caltech10_c4 --algorithm fdse --gpu 0 --load_mode mmap --config ./config/office/fdse.yml --logger PerRunLogger --seed 2

python run_single.py --task PACS_c4 --algorithm fdse --gpu 0 --load_mode mmap --config ./config/pacs/fdse.yml --logger PerRunLogger --seed 2
```

You can also use `run_seed.py` to run multiple random seeds in parallel to obtain the main results in the paper.

The console will output log like
```shell
...
2026-01-14 09:55:28,871 fedbase.py run [line:311] INFO --------------Round 122--------------
2026-01-14 09:55:29,978 __init__.py log_once [line:203] INFO local_val_accuracy            0.7788
2026-01-14 09:55:29,978 __init__.py log_once [line:203] INFO mean_local_val_accuracy       0.7616
2026-01-14 09:55:29,978 __init__.py log_once [line:203] INFO std_local_val_accuracy        0.0822
2026-01-14 09:55:29,978 __init__.py log_once [line:203] INFO min_local_val_accuracy        0.6293
2026-01-14 09:55:29,978 __init__.py log_once [line:203] INFO max_local_val_accuracy        0.8550
2026-01-14 09:55:29,978 __init__.py log_once [line:203] INFO local_val_loss                0.7603
2026-01-14 09:55:29,978 __init__.py log_once [line:203] INFO mean_local_val_loss           0.8010
2026-01-14 09:55:29,978 __init__.py log_once [line:203] INFO std_local_val_loss            0.2955
2026-01-14 09:55:29,978 __init__.py log_once [line:203] INFO min_local_val_loss            0.5388
2026-01-14 09:55:29,978 __init__.py log_once [line:203] INFO max_local_val_loss            1.2813
2026-01-14 09:55:29,978 __init__.py log_once [line:203] INFO local_test_accuracy           0.8335
2026-01-14 09:55:29,978 __init__.py log_once [line:203] INFO mean_local_test_accuracy      0.8136
2026-01-14 09:55:29,978 __init__.py log_once [line:203] INFO std_local_test_accuracy       0.0709
2026-01-14 09:55:29,978 __init__.py log_once [line:203] INFO min_local_test_accuracy       0.7157
2026-01-14 09:55:29,979 __init__.py log_once [line:203] INFO max_local_test_accuracy       0.9107
2026-01-14 09:55:29,979 __init__.py log_once [line:203] INFO local_test_loss               0.5634
2026-01-14 09:55:29,979 __init__.py log_once [line:203] INFO mean_local_test_loss          0.6390
2026-01-14 09:55:29,979 __init__.py log_once [line:203] INFO std_local_test_loss           0.2585
2026-01-14 09:55:29,979 __init__.py log_once [line:203] INFO min_local_test_loss           0.2759
2026-01-14 09:55:29,979 __init__.py log_once [line:203] INFO max_local_test_loss           0.9745
2026-01-14 09:55:29,979 fedbase.py run [line:316] INFO Eval Time Cost:               1.1076s
```
# Citation
```
@misc{wang2025federatedlearningdomainshift,
      title={Federated Learning with Domain Shift Eraser}, 
      author={Zheng Wang and Zihui Wang and Zheng Wang and Xiaoliang Fan and Cheng Wang},
      year={2025},
      eprint={2503.13063},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.13063}, 
}
```
