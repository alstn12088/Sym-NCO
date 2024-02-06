import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn import CosineSimilarity
from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost,_ = rollout(model, dataset, opts)
    avg_cost = cost.mean()

    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
  
        

        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))

        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0),None


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()
 


    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            opts,problem
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, opts)

 


    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()



###################################################################### 
# Problem Symmetric Transformation
###################################################################### 

def random_data_augmentation(batch, validate=False):

    x = batch[:, :, [0]]
    y = batch[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)
    
    data_list = [dat1,dat2,dat3,dat4,dat5,dat6,dat7,dat8]
    
    

    if validate:
        index = torch.randperm(7)[:1] + 1
        batch1 = dat1
        batch2 = data_list[index[0]]
        return batch1, batch2
    index = torch.randperm(8)[:2]
    batch1 = data_list[index[0]]
    batch2 = data_list[index[1]]

    return batch1,batch2


def rotational_loss(model, batch):

    _, batch_aug2 = random_data_augmentation(batch, validate=False)
    model1 = model.clone()
    model1.eval() 
    with torch.no_grad():
        cost,_ = model1(batch)



    return cost


def SR_transform(x, y, idx):
    if idx < 0.5:
        phi = idx * 4 * math.pi
    else:
        phi = (idx - 0.5) * 4 * math.pi

    x = x - 1 / 2
    y = y - 1 / 2

    x_prime = torch.cos(phi) * x - torch.sin(phi) * y
    y_prime = torch.sin(phi) * x + torch.cos(phi) * y

    if idx < 0.5:
        dat = torch.cat((x_prime + 1 / 2, y_prime + 1 / 2), dim=2)
    else:
        dat = torch.cat((y_prime + 1 / 2, x_prime + 1 / 2), dim=2)
    return dat


def augment_xy_data_by_N_fold(problems, N, depot=None):
    x = problems[:, :, [0]]
    y = problems[:, :, [1]]

    if depot is not None:
        x_depot = depot[:, :, [0]]
        y_depot = depot[:, :, [1]]
    idx = torch.rand(N - 1)

    for i in range(N - 1):

        problems = torch.cat((problems, SR_transform(x, y, idx[i])), dim=0)
        if depot is not None:
            depot = torch.cat((depot, SR_transform(x_depot, y_depot, idx[i])), dim=0)

    if depot is not None:
        return problems, depot.view(-1, 2)

    return problems


def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems


def augment(input, N,problem):
    is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
    is_orienteering = problem.NAME == 'op'
    is_pctsp = problem.NAME == 'pctsp'
    if is_vrp or is_orienteering or is_pctsp:
        if is_vrp:
            features = ('demand',)
        elif is_orienteering:
            features = ('prize','max_length')
        else:
            assert is_pctsp
            features = ('deterministic_prize', 'penalty')

        input['loc'], input['depot'] = augment_xy_data_by_N_fold(input['loc'], N, depot=input['depot'].view(-1, 1, 2))

        for feat in features:
            input[feat] = input[feat].repeat(N, 1)
        if is_orienteering:
            input['max_length'] = input['max_length'].view(-1)
        return input

        # TSP
    return augment_xy_data_by_N_fold(input, N)

###################################################################### 
# Problem Symmetric Transformation
###################################################################### 




def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        opts,problem
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities


    x_aug = augment(x,opts.N_aug,problem)

    cost, log_likelihood, proj_nodes = model(x_aug,return_proj=True)

    ###################################################################### 
    # rotational invariant consistancy learning
    ###################################################################### 
    proj_nodes = proj_nodes.reshape(opts.N_aug, -1, proj_nodes.shape[1], proj_nodes.shape[2])
    
    cos = torch.nn.CosineSimilarity(dim=-1)
    similarity = 0
    for i in range(opts.N_aug - 1):
        similarity = similarity + cos(proj_nodes[0], proj_nodes[i + 1])

    similarity /= (opts.N_aug - 1)
    ###################################################################### 
    # rotational invariant consistancy learning
    ######################################################################

    cost = cost.view(opts.N_aug,-1).permute(1,0)
    log_likelihood = log_likelihood.view(opts.N_aug,-1).permute(1,0)
    
    # problem symmetric loss
    advantage = cost - cost.mean(dim=1).view(-1,1)


    loss = ((advantage) * log_likelihood).mean() - opts.alpha * similarity.mean()
    

    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

