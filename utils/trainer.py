import logging
import os
import pdb

from tqdm import tqdm 
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data

from utils.helper import RunningAverage, save_checkpoint, load_checkpoint, get_logger, get_batch_size
from utils.visualize import VisdomLinePlotter

from models.gcn.losses import get_loss_criterion
from models.gcn.metrics import get_evaluation_metric

class Trainer:
    def __init__(self, model, optimizer, lr_scheduler,
                   device, loaders, checkpoint_dir, max_num_epochs=1000,
                   num_iterations=0, num_epoch=0,
                   eval_score_higher_is_better=True, best_eval_score=None,
                   logger=None):
        if logger is None:
            self.logger = get_logger('Trainer', level=logging.DEBUG)
        else:
            self.logger = logger
        self.plotter = VisdomLinePlotter('gcn')

        self.logger.info(model)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.eval_score_higher_is_better = eval_score_higher_is_better
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')
        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))
        self.num_iterations = num_iterations
        self.num_epoch = num_epoch

    def fit(self):
        # pretrain encoder and classifier
        self.model.train()
        N_pre = 40
        for i in range(N_pre):
            self.logger.info(f'Pretraining epoch [{i}/{N_pre - 1}]. ')
            for _, t in enumerate(self.loaders['train']):
                target = t.y[:, 0].to(self.device)
                input = t.to(self.device)
                S = self.loaders['S']
                S = [S[0].to(self.device), S[1].to(self.device), S[2], S[3]]
                conn_est, output, _, _, _, _ = self.model(input, S)
                conn = input.conn.view(input.num_graphs, int(input.num_nodes/input.num_graphs), -1).clone().to(dtype=torch.float32)
                #value, _ = torch.topk(conn_est.view(input.num_graphs, -1), int(conn.size(1)**2 * S[3]/100), dim=1)
                #mask = conn_est > value[:, -1][:, None, None]
                #conn_est = conn_est*mask.float()

                # compute loss criterion
                loss = get_loss_criterion('BCELoss')(conn_est, conn)
                #loss += get_loss_criterion('CrossEntropyLoss')(output, target)

                # compute gradients and update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
        for _ in range(self.max_num_epochs):
            # train for one epoch
            self.train(self.loaders['train'], self.loaders['S'])
            self.num_epoch += 1

        return self.model

    def train(self, train_loader, S):
        train_losses = RunningAverage()
        train_eval_scores = RunningAverage()
        train_reconn_losses = RunningAverage()

        self.logger.info(
            f'Training epoch [{self.num_epoch}/{self.max_num_epochs - 1}], iteration per epoch: {len(train_loader)}. ')
        S = [S[0].to(self.device), S[1].to(self.device), S[2], S[3]]

        # sets the model in training mode
        self.model.train()
        for i, t in enumerate(train_loader):
            target = t.y[:, 0].to(self.device)
            input = t.to(self.device)
            conn_est, output, h_mean, h_var, _, _ = self.model(input, S)
            conn = input.conn.view(input.num_graphs, int(input.num_nodes/input.num_graphs), -1).clone().to(dtype=torch.float32)
            #value, _ = torch.topk(conn_est.view(input.num_graphs, -1), int(conn.size(1)**2 * S[3]/100), dim=1)
            #mask = conn_est > value[:, -1][:, None, None]
            #conn_est = conn_est*mask.float()

            # compute loss criterion
            loss_classifier = get_loss_criterion('MSEClassLoss')(output, target)
            loss_KLD_h = get_loss_criterion('GaussianKLDLoss')(h_var, h_mean)
            loss_reconn = get_loss_criterion('BCELoss')(conn_est, conn)
            loss = 2*loss_classifier + loss_KLD_h + loss_reconn
            #loss = loss_classifier + loss_reconn
            train_losses.update(loss.item(), get_batch_size(target))
            train_reconn_losses.update(loss_reconn.item(), get_batch_size(target))

            # compute eval criterion
            eval_score = get_evaluation_metric('ClassMSE')(output, target)
            train_eval_scores.update(eval_score.item(), get_batch_size(target))

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.num_iterations += 1

        max_num_iterations = self.max_num_epochs * len(train_loader)
        self.logger.info(f'Training iteration [{self.num_iterations}/{max_num_iterations - 1}]. Batch [{i}/{len(train_loader) - 1}]. Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')
        self.logger.info(f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')
        self._log_stats('train', train_losses.avg, train_eval_scores.avg)
        self.plotter.plot('loss', 'train', 'loss', self.num_iterations, train_losses.avg, xlabel='Iter')
        self.plotter.plot('accuracy', 'train', 'accuracy', self.num_iterations, train_eval_scores.avg, xlabel='Iter')
        self.plotter.plot('loss_reconn', 'train', 'loss_reconn', self.num_iterations, train_reconn_losses.avg, xlabel='Iter')

        # evaluate on validation set
        eval_score = self.validate(self.loaders['val'], S)
        # adjust learning rate if necessary
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(eval_score)
        else:
            self.scheduler.step()
        # log current learning rate in tensorboard
        self._log_lr()
        # remember best validation metric
        is_best = self._is_best_eval_score(eval_score)
        # save checkpoint
        self._save_checkpoint(is_best)
        #self._log_params()

    def validate(self, val_loader, S):
        val_scores = RunningAverage()
        val_reconn_losses = RunningAverage()

        self.logger.info(f'Validating epoch [{self.num_epoch}/{self.max_num_epochs - 1}]. ')
        
        try:
            self.model.eval()
            with torch.no_grad():
                for _, t in enumerate(val_loader):
                    target = t.y[:, 0].to(self.device)
                    input = t.to(self.device)

                    conn_est, output, _, _, _, _ = self.model(input, S)
                    conn = input.conn.view(input.num_graphs, int(input.num_nodes/input.num_graphs), -1).clone().to(dtype=torch.float32)
                    #value, _ = torch.topk(conn_est.view(input.num_graphs, -1), int(conn.size(1)**2 * S[3]/100), dim=1)
                    #mask = conn_est > value[:, -1][:, None, None]
                    #conn_est = conn_est*mask.float()

                    # compute loss criterion
                    loss_reconn = get_loss_criterion('BCELoss')(conn_est, conn)
                    val_reconn_losses.update(loss_reconn.item(), get_batch_size(target))

                    # compute eval criterion
                    eval_score = get_evaluation_metric('ClassMSE')(output, target)
                    val_scores.update(eval_score.item(), get_batch_size(target))

                self._log_stats('val', 0, val_scores.avg)
                self.logger.info(f'Validation finished. Evaluation score: {val_scores.avg}')
                self.logger.info(f'--------------------------------------------------------------------')
                self.plotter.plot('loss_reconn', 'val', 'loss_reconn', self.num_iterations, val_reconn_losses.avg, xlabel='Iter')
                self.plotter.plot('accuracy', 'val', 'accuracy', self.num_iterations, val_scores.avg, xlabel='Iter')

                return val_scores.avg
        finally:
            # set back in training mode
            self.model.train()
    
    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score >= self.best_eval_score
        else:
            is_best = eval_score <= self.best_eval_score

        if is_best:
            self.logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score
        return is_best

    def _save_checkpoint(self, is_best):
        #if torch.cuda.device_count() > 1:
        #    model_state = self.model.module.state_dict()
        #else:
        model_state = self.model.state_dict()
        save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': model_state,
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs}, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=self.logger)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        self.logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)