# Sym-NCO: Leveraging Symmetricity for Neural Combinatorial Optimization 

> ## 🔔 Note (Updated Implementation)
>
> A more **extensive, actively maintained, and up-to-date implementation** of Neural Combinatorial Optimization (including Sym-NCO–related components and broader RL4CO methods) is now available in the following repository:
>
> 👉 https://github.com/ai4co/rl4co
>
> We recommend users check this repository for the latest features, improvements, and unified implementations of recent neural combinatorial optimization methods.
>
> This repository remains the **official code release for the NeurIPS 2022 Sym-NCO paper** and is preserved for reproducibility of the original experiments.


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
python eval.py --dataset_path 'data/pctsp100_test_seed1234.pkl' --model pretrained_model/pctsp_100/epoch-99.pt 
```

**OP reproduce**
```bash
python eval.py --dataset_path 'data/op_dist100_test_seed1234.pkl' --model pretrained_model/op_100/epoch-99.pt 
```

### Train

**General**
```bash
python run.py --problem [Target Problem ('tsp', 'cvrp', 'pctsp_det', 'op')] --N_aug [L: problem symmetric width]
```

**Example**

```bash
python run.py --problem 'tsp' --N_aug 10 
```

## Dependencies (Same with AM)

* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.7
* tqdm
* pytz
* Matplotlib (optional, only for plotting)


## Further Work

Symmetric learning gives powerful benefits for combinatorial optimization. However, there are no remaining rooms at DRL. Instead, imitation learning with sparse data setting can be an alterative benchmark for further work. 
If you want to do further work of Sym-NCO, please check out [Sym-NCO-IL](https://github.com/alstn12088/Sym-NCO-IL). 



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

