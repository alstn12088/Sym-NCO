# Sym-NCO: Leveraging Symmetricity for Neural Combinatorial Optimization 

Sym-NCO is deep reinforcement learning-based neural combinatorial optimization scheme that exploits the symmetric nature of combinatorial optimization. 

Before reading our code, we strongly recommend reading the code of [AM](https://github.com/wouterkool/attention-learn-to-route) and [POMO](https://github.com/yd-kwon/POMO/tree/master/NEW_py_ver) which is base of our code. 


## Sym-NCO-POMO

Sym-NCO is an extended method of POMO. POMO will give powerful performances in TSP. Sym-NCO can improve POMO at CVRP and slightly at TSP. 

### TSP

Firstly, go to folder:
```bash
cd Sym-NCO-POMO/TSP/
```

#### Test




**Sym-NCO test**
```bash
python test_symnco.py
```

You can change "test_batch_size", "aug_factor" (related to sample width), "aug_batch_size". 

**POMO baseline test**
```bash
python test_baseline.py
```


#### Training

**Sym-NCO training**
```bash
python train_symnco.py
```

**POMO training**
```bash
python train_baseline.py
```

### CVRP

Firstly, go to folder:
```bash
cd Sym-NCO-POMO/CVRP/
```

#### Test




**Sym-NCO test**
```bash
python test_symnco.py
```

You can change "test_batch_size", "aug_factor" (related to sample width), "aug_batch_size". 

**POMO baseline test**
```bash
python test_baseline.py
```


#### Training

**Sym-NCO training**
```bash
python train_symnco.py
```

**POMO training**
```bash
python train_baseline.py
```

## Sym-NCO-AM

Sym-NCO can be also applied to vanilla AM model. 

AM is more expandable to solve various problems including TSP,CVRP,PCTSP and OP.

We provide pretrained Sym-NCO based AM model for PCTSP and OP. 


Firstly, go to folder:
```bash
cd Sym-NCO-AM/
```

### Test

**General**
```bash
python eval.py --dataset_path [YOUR_DATASET] --model [YOUR_MODEL] --eval_batch_size [YOUR BATCH SIZE] -- augment [SAMPLE WIDTH]
```

**PCTSP reproduce**
```bash
python eval.py --dataset_path 'pctsp100_test_seed1234.pkl' --model pretrained_model/pctsp_100/epoch-99.pt 
```

**OP reproduce**
```bash
python eval.py --dataset_path 'op_dist100_test_seed1234.pkl' --model pretrained_model/op_100/epoch-99.pt 
```

### Train

**General**
```bash
python train.py --problem [Target Problem ('tsp', 'cvrp', 'pctsp_det', 'op')] --N_aug [L: problem symmetric width]
```

**Example**

```bash
python train.py --problem 'tsp' --N_aug 10 
```

## Dependencies (Same with AM)

* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.7
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib (optional, only for plotting)


## Further Work

Symmetric learning gives powerful benefit for combinatorial optimization. However, I think there is no remaining rooms at DRL. Instead, we think imitation learning with sparse data setting can be alterative benchmark for further work. 
If you want to make further work of Sym-NCO, please check out [Sym-NCO-IL](https://github.com/alstn12088/Sym-NCO-IL). 



## Paper
This is official PyTorch code for our paper [Sym-NCO: Leveraging Symmetricity for Neural Combinatorial Optimization](https://openreview.net/forum?id=kHrE2vi5Rvs) which has been accepted at [NeurIPS 2022].

```
@article{kim2022sym,
  title={Sym-NCO: Leveraging Symmetricity for Neural Combinatorial Optimization},
  author={Kim, Minsu and Park, Junyoung and Park, Jinkyoo},
  journal={arXiv preprint arXiv:2205.13209},
  year={2022}
}
```

