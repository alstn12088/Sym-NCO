import torch
from logging import getLogger

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model
from TSPModel_ours import TSPModel as Model_ours
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
import torch.nn.functional as F
from utils.utils import *

class TSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params,tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.tester_params = tester_params
        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components

        if self.trainer_params['is_pomo']:
            self.model = Model(**self.model_params)
        else:
            self.model = Model_ours(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        # if self.trainer_params['wandb']:
        #     import wandb
        #     wandb.init(project="tsp_ablation_50", entity="alstn12088")
        #     self.wandb = wandb

        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)
            a,b,c = self._test()

            # if self.trainer_params['wandb']:
            #     self.wandb.log({"greedy": a})
            #     self.wandb.log({"pomo": b})
            #     self.wandb.log({"pomo_aug": c})


    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):

    # Prep
    ###############################################
        self.model.train()
        self.env.load_problems(batch_size, self.env_params['sr_size'])
        reset_state, _, _ = self.env.reset()
        proj_nodes = self.model.pre_forward(reset_state,return_h_mean=True)
    

        prob_list = torch.zeros(size=(batch_size*self.env_params['sr_size'], self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        i=0
        while not done:


            selected, prob = self.model(state=state)
            # if i==1:
            #     entropy = -prob * torch.log(prob)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            i = i + 1
    #prob_list = prob_list.reshape(self.env_params['sr_size'],batch_size,self.env.pomo_size, -1).permute(1,0,2,3).reshape(batch_size,self.env.pomo_size*self.env_params['sr_size'],-1)
    #reward = reward.reshape(self.env_params['sr_size'],batch_size, self.env.pomo_size).permute(1,0,2).reshape(batch_size,self.env.pomo_size*self.env_params['sr_size'])
    #entropy = entropy.reshape(self.env_params['sr_size'],batch_size, self.env.pomo_size).permute(1,0,2).reshape(batch_size,self.env.pomo_size*self.env_params['sr_size'])


        # ours
        if self.env_params['sr_size']>1:

            # State Invariant
            ###############################################
            proj_nodes = proj_nodes.reshape(self.env_params['sr_size'], batch_size, -1)

            proj_nodes = F.normalize(proj_nodes, dim=-1)

            proj_1 = proj_nodes[0]
            proj_2 = proj_nodes[1]

            similarity_matrix = torch.matmul(proj_1, proj_2.T)
            mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool)
            positive = similarity_matrix[mask].view(similarity_matrix.shape[0],-1)
            negative = similarity_matrix[~mask].view(similarity_matrix.shape[0],-1)


            negative = torch.exp(negative).sum(dim=-1,keepdim=True)

            sim_loss = -(positive - torch.log(negative)).mean()



            #cos = torch.nn.CosineSimilarity(dim=-1)
            #similarity = 0
            #for i in range(self.env_params['sr_size']-1):
                #similarity = similarity + cos(proj_nodes[0],proj_nodes[i+1])

            #similarity /= (self.env_params['sr_size']-1)

       
            # State Symmetricity
            ###############################################

            prob_list_sr \
                = prob_list.view(self.env_params['sr_size'], batch_size, self.env.pomo_size, -1).permute(1, 2, 0,3).reshape(batch_size,self.env_params['sr_size']*self.env.pomo_size,-1)
            reward_sr \
                = reward.view(self.env_params['sr_size'], batch_size, self.env.pomo_size).permute(1, 2, 0).reshape(batch_size,self.env_params['sr_size']*self.env.pomo_size)


            # shape: (batch,pomo,sr_size)
            advantage_sr = reward_sr - reward_sr.float().mean(dim=1,keepdims=True)

            # shape: (batch,pomo,sr_size)W
            log_prob_sr = prob_list_sr.log().sum(dim=2)
            loss_sr = -advantage_sr*log_prob_sr
            loss_sr_mean = loss_sr.mean()

            # Action (pomo) Symmetricity
            ###############################################
            prob_list_pomo \
                = prob_list.view(self.env_params['sr_size'], batch_size, self.env.pomo_size, -1)[0]
            reward_pomo \
                = reward.view(self.env_params['sr_size'], batch_size, self.env.pomo_size)[0]
            # shape: (batch,sr_size,pomo)
            advantage_pomo = reward_pomo - reward_pomo.float().mean(dim=1, keepdims=True)

            # shape: (batch,sr_size,pomo)
            log_prob_pomo = prob_list_pomo.log().sum(dim=2)
            loss_pomo = -advantage_pomo * log_prob_pomo
            loss_pomo_mean = loss_pomo.mean()
            # if self.trainer_params['wandb']:
            #     self.wandb.log({"sim_loss": sim_loss})
            #     #self.wandb.log({"similarity": similarity.mean()})
            #     self.wandb.log({"reward": reward.mean()})
            # Sum of two symmetric loss
            #loss_mean = loss_pomo_mean + loss_sr_mean - self.trainer_params['alpha'] * similarity.mean()
            loss_mean = loss_pomo_mean + loss_sr_mean + self.trainer_params['alpha'] * sim_loss

            reward \
                = reward.reshape(self.env_params['sr_size'],batch_size, self.env.pomo_size).permute(1,0,2).reshape(batch_size,self.env.pomo_size*self.env_params['sr_size'])

        else:


            proj_nodes = proj_nodes.reshape(self.env_params['sr_size'], batch_size, proj_nodes.shape[1],-1)
            cos = torch.nn.CosineSimilarity(dim=-1)


            similarity = cos(proj_nodes[0],proj_nodes[0])
            # if self.trainer_params['wandb']:
            #     self.wandb.log({"similarity": similarity.mean()})
            #     self.wandb.log({"reward": reward.mean()})
            # Loss
            ###############################################
            advantage = reward - reward.float().mean(dim=1, keepdims=True)
            # shape: (batch, pomo)
            log_prob = prob_list.log().sum(dim=2)
            # size = (batch, pomo)
            loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
            # shape: (batch, pomo)
            loss_mean = loss.mean()


        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item()

    def _test(self):

        no_pomo_score_AM = AverageMeter()
        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()


        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            no_pomo_score,score, aug_score = self._test_one_batch(batch_size)

            no_pomo_score_AM.update(no_pomo_score, batch_size)
            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size
        return no_pomo_score_AM.avg, score_AM.avg, aug_score_AM.avg


    def _test_one_batch(self, batch_size):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

  

        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        no_pomo_score = -aug_reward[0, :, 0].mean()

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_pomo_score.item(), no_aug_score.item(), aug_score.item()
