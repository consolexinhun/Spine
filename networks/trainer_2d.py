import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from . import utils


class Trainer:
    """

    Args:
        model: UNet 2D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
    """

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 eval_criterion, device, train_loader, val_loader, checkpoint_dir,
                 max_num_epochs=100, batch_size=2, max_num_iterations=1e5,
                 validate_after_iters=100, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=True, best_eval_score=None,
                 logger=None,
                 ds_weight=0):
        if logger is None:
            self.logger = utils.get_logger('ConvCRFTrainer', level=logging.DEBUG)
        else:
            self.logger = logger

        self.logger.info(model)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.batch_size = batch_size
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.ds_weight = ds_weight
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

    @classmethod
    def from_checkpoint(cls, checkpoint_path, model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                        train_loader, val_loader,
                        logger=None, ds_weight=0):
        logger.info(f"Loading checkpoint '{checkpoint_path}'...")
        state = utils.load_checkpoint(checkpoint_path, model, optimizer)
        logger.info(
            f"Checkpoint loaded. Epoch: {state['epoch']}. Best val score: {state['best_eval_score']}. Num_iterations: {state['num_iterations']}")
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   torch.device(state['device']),
                   train_loader, val_loader, checkpoint_dir,
                   eval_score_higher_is_better=state['eval_score_higher_is_better'],
                   best_eval_score=state['best_eval_score'],
                   num_iterations=state['num_iterations'],
                   num_epoch=state['epoch'],
                   batch_size=state['batch_size'],
                   max_num_epochs=state['max_num_epochs'],
                   max_num_iterations=state['max_num_iterations'],
                   validate_after_iters=state['validate_after_iters'],
                   log_after_iters=state['log_after_iters'],
                   validate_iters=state['validate_iters'],
                   logger=logger,
                   ds_weight=ds_weight)

    @classmethod
    def from_pretrained(cls, pre_trained, model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                        device, train_loader, val_loader,
                        max_num_epochs=100, batch_size=2, max_num_iterations=1e5,
                        validate_after_iters=100, log_after_iters=100,
                        validate_iters=None, num_iterations=1, num_epoch=0,
                        eval_score_higher_is_better=True, best_eval_score=None,
                        logger=None,
                        ds_weight=0):
        logger.info(f"Logging pre-trained model from '{pre_trained}'...")
        utils.load_checkpoint(pre_trained, model, None)
        checkpoint_dir = os.path.split(pre_trained)[0]
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   device, train_loader, val_loader, checkpoint_dir,
                   eval_score_higher_is_better=eval_score_higher_is_better,
                   best_eval_score=best_eval_score,
                   num_iterations=num_iterations,
                   num_epoch=num_epoch,
                   max_num_epochs=max_num_epochs,
                   batch_size=batch_size,
                   max_num_iterations=max_num_iterations,
                   validate_after_iters=validate_after_iters,
                   log_after_iters=log_after_iters,
                   validate_iters=validate_iters,
                   logger=logger,
                   ds_weight=ds_weight)

    def fit(self):
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train(self.train_loader)

            if should_terminate:
                break

            self.num_epoch += 1
            self.scheduler.step()

    def train(self, train_loader):
        """Trains the model for 1 epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): training data loader

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = utils.RunningAverage()
        train_eval_scores = utils.RunningAverage()

        # sets the model in training mode
        self.model.train()

        # Call the function of train_loader.__iter__ to return a batch. Here, the train_loader is an instance of DataLoader
        for i, t in enumerate(train_loader):
            self.logger.info(
                f'Training iteration {self.num_iterations}. Batch {i}. Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')

            mr, weight, unary, target = self._split_training_batch(t)
            if mr.shape[0] < self.batch_size:
                continue

            output, loss = self._forward_pass(mr=mr, unary=unary, mask=target, weight=weight)
            print('loss: %.4f, batch_size: %d' % (loss.item(), self._batch_size(mr)))
            train_losses.update(loss.item(), self._batch_size(mr))

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.num_iterations % self.validate_after_iters == 0:
                # evaluate on validation set
                eval_score = self.validate(self.val_loader)
                # adjust learning rate if necessary
                # if isinstance(self.scheduler, ReduceLROnPlateau):
                #     self.scheduler.step(eval_score)
                # else:
                #     self.scheduler.step()
                # log current learning rate in tensorboard
                self._log_lr()
                # remember best validation metric
                is_best = self._is_best_eval_score(eval_score)

                # save checkpoint
                self._save_checkpoint(is_best)

            if self.num_iterations % self.log_after_iters == 0:
                output = nn.Softmax(dim=1)(output)

                # compute eval criterion
                batch_size = output.shape[0]
                if batch_size > 1:
                    eval_scores = []
                    for j in range(0, batch_size):
                        eval_scores.append(self.eval_criterion(output[j].unsqueeze(0), target[j].unsqueeze(0)))
                    eval_score = np.mean(eval_scores)
                    train_eval_scores.update(eval_score, self._batch_size(mr))
                else:
                    eval_score = self.eval_criterion(output, target)
                    train_eval_scores.update(eval_score.item(), self._batch_size(mr))

                # log stats, params and images
                self.logger.info(
                    f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')
                self._log_stats('train', train_losses.avg, train_eval_scores.avg)
                self._log_params()
                self._log_images(mr, target, output)

            if self.max_num_iterations < self.num_iterations:
                self.logger.info(
                    f'Maximum number of iterations {self.max_num_iterations} exceeded. Finishing training...')
                return True

            self.num_iterations += 1

        return False

    def validate(self, val_loader):
        self.logger.info('Validating...')

        val_losses = utils.RunningAverage()
        val_scores = utils.RunningAverage()

        try:
            self.model.eval()
            with torch.no_grad():
                for i, t in enumerate(val_loader):
                    self.logger.info(f'Validation iteration {i}')

                    mr, weight, unary, target = self._split_training_batch(t)

                    output, loss = self._forward_pass(mr=mr, unary=unary, mask=target, weight=weight)
                    val_losses.update(loss.item(), self._batch_size(mr))

                    # compute eval criterion
                    batch_size = output.shape[0]
                    if batch_size > 1:
                        eval_scores = []
                        for j in range(0, batch_size):
                            eval_scores.append(self.eval_criterion(output[j].unsqueeze(0), target[j].unsqueeze(0)))
                        eval_score = np.mean(eval_scores)
                        val_scores.update(eval_score, self._batch_size(mr))
                    else:
                        eval_score = self.eval_criterion(output, target)
                        val_scores.update(eval_score.item(), self._batch_size(mr))

                    if self.validate_iters is not None and self.validate_iters <= i:
                        # stop validation
                        break

                self._log_stats('val', val_losses.avg, val_scores.avg)
                self.logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
                return val_scores.avg
        finally:
            # set back in training mode
            self.model.train()

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                return input.to(self.device)

        t = _move_to_device(t)
        if len(t) == 4:
            mr, weight, unary, mask = t
        else:
            mr, unary, mask = t
            weight = None
        return mr, weight, unary, mask

    def _forward_pass(self, mr,  unary, mask, weight=None, feature=None):
        # forward pass
        output = self.model(unary=unary, img=mr, col_feats=feature)
        if isinstance(output, tuple) or isinstance(output, list):
            if len(output) == 3:
                # for test
                # final_activation, final_conv, fea_logit = output
                # compute the loss
                if weight is None:
                    loss1 = self.loss_criterion(output[1], mask)
                else:
                    loss1 = self.loss_criterion(output[1], mask, weight)
                if output[2] is not None:
                    up_fea_logit = F.interpolate(output[2], size=mask.size()[1:], mode='bilinear',
                                                 align_corners=True)
                    if weight is None:
                        loss2 = self.loss_criterion(up_fea_logit, mask)
                    else:
                        loss2 = self.loss_criterion(up_fea_logit, mask, weight)
                    loss = loss1 + self.ds_weight * loss2
                else:
                    loss = loss1
                return output[0], loss
            elif len(output) == 2:
                # for training
                # final_conv, fea_logit = output
                if weight is None:
                    loss1 = self.loss_criterion(output[0], mask)
                else:
                    loss1 = self.loss_criterion(output[0], mask, weight)
                if output[1] is not None:
                    up_fea_logit = F.interpolate(output[1], size=mask.size()[1:], mode='bilinear', align_corners=True)
                    if weight is None:
                        loss2 = self.loss_criterion(up_fea_logit, mask)
                    else:
                        loss2 = self.loss_criterion(up_fea_logit, mask, weight)
                    loss = loss1 + self.ds_weight * loss2
                else:
                    loss = loss1
                return output[0], loss
        else:
            # compute the loss
            if weight is None:
                loss = self.loss_criterion(output, mask)
            else:
                loss = self.loss_criterion(output, mask, weight)

            return output, loss

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            self.logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):
        utils.save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': self.model.state_dict(),
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=self.logger)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        print('learning_rate = %f' % lr)
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
            if value.grad is not None:
                self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction):
        inputs_map = {
            'inputs': input,
            'targets': target,
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self._images_from_batch(name, batch):
                self.writer.add_image(tag, image, self.num_iterations, dataformats='HW')

    def _images_from_batch(self, name, batch):
        if batch.ndim == 5:
            # NCDHW
            tag_template = '{}/batch_{}/channel_{}/slice_{}'

            tagged_images = []
            slice_idx = batch.shape[2] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                for channel_idx in range(batch.shape[1]):
                    tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)
                    img = batch[batch_idx, channel_idx, slice_idx, ...]
                    tagged_images.append((tag, self._normalize_img(img)))

        # NCHW
        elif batch.ndim == 4:
            tag_template = '{}/batch_{}/channel_{}'

            tagged_images = []
            for batch_idx in range(batch.shape[0]):
                for channel_idx in range(batch.shape[1]):
                    tag = tag_template.format(name, batch_idx, channel_idx)
                    img = batch[batch_idx, channel_idx, :, :]
                    tagged_images.append((tag, self._normalize_img(img)))
        else: # NHW
            tag_template = '{}/batch_{}/channel_{}'

            tagged_images = []
            for batch_idx in range(batch.shape[0]):
                tag = tag_template.format(name, batch_idx, 0)
                img = batch[batch_idx, :, :]
                tagged_images.append((tag, self._normalize_img(img)))


        return tagged_images

    @staticmethod
    def _normalize_img(img):
        return (img - np.min(img)) / (np.ptp(img) + 1e-7)

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
