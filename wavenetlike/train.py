# train.py
"""
    training loops
"""
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import logging

import wavenetlike.ops as ops


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self,
                 model,
                 dataset,
                 optimizer=torch.optim.Adam,
                 learning_rate=0.0001,
                 weight_decay=0.0,
                 nepochs=10**5,
                 epoch_save_freq=100):

        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer(
            params=self.model.parameters(),
            lr=learning_rate
        )
        self.dataloader = None
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.nepochs = nepochs
        self.epoch_save_freq = epoch_save_freq

        self.writer = SummaryWriter()

        self.dtype = torch.FloatTensor
        self.ltype = torch.LongTensor

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            print('using gpu')
            self.dtype = torch.cuda.FloatTensor
            self.ltype = torch.cuda.LongTensor
        else:
            print('using cpu')

        if use_cuda:
            print('moving model to gpu')
            self.model.cuda()

    def train(self):

        train_artifacts_dir = os.path.join(os.curdir, 'trainartifacts')
        if not os.path.exists(train_artifacts_dir):
            os.mkdir(train_artifacts_dir)

        checkpt_path = os.path.join(train_artifacts_dir, "checkpoint")
        model_save_path = os.path.join(train_artifacts_dir, "model")

        stepOp = WavenetStepOp(self.model.in_out_channel_size,
                               self.model.receptive_field_size)

        saved_epoch = 0

        logger.info("checking for existing checkpointed model ...")
        if os.path.exists(checkpt_path):
            checkpoint = torch.load(checkpt_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            saved_epoch = checkpoint['epoch']
            # saved_epoch_loss = checkpoint['loss']
            logger.info(f"loaded checkpoint model at epoch {saved_epoch}")
        else:
            logger.info("no checkpointed model found.")

        logger.info("training ...")

        j = 0
        for epoch in range(self.nepochs):
            epoch_loss = 0.
            n_samples = 0.
            for i, audio_sample in enumerate(self.dataset):
                loss_value = stepOp.step(self.optimizer, self.model, audio_sample, j,
                                         self.writer)
                epoch_loss += loss_value
                n_samples += 1
                j += 1
            epoch_loss /= n_samples
            logger.info(f'(epoch = {epoch}) epoch loss: {epoch_loss}')

            self.writer.add_scalar("Loss/train", epoch_loss, epoch)
            self.writer.flush()

            if epoch > self.nepochs:
                break
            if epoch % self.epoch_save_freq == 0:
                torch.save({
                    'epoch': epoch + saved_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': epoch_loss
                }, checkpt_path
                )
                logger.info(f"saved checkpoint at epoch {epoch + saved_epoch}")
                logger.info(f'saving model to {model_save_path}...')
                torch.save(self.model.state_dict(), model_save_path)


class WavenetStepOp:

    def __init__(self, audio_channel_size, receptive_field_size):
        self.audio_channel_size = audio_channel_size
        self.receptive_field_size = receptive_field_size

    def step(self, optimizer, model, audio_sample, step_j, writer=None):
        waveform, sample_rate, labels1, labels2, labels3 = audio_sample
        x = ops.waveform_to_input(waveform,
                                  m=self.audio_channel_size)
        cx = ops.waveform_to_categorical(waveform,
                                         m=self.audio_channel_size)
        # logger.info(f'input shape: {x.shape}')
        y = model(x)
        # logger.info(f'output shape: {y.shape}')

        optimizer.zero_grad()
        loss = ops.softmax_loss_fn(y, cx, self.receptive_field_size,
                                   show_match_fraction=True)
        fmatched = ops.match_fraction(y, cx, self.receptive_field_size)
        if writer:
            writer.add_scalar("Fraction matched/train", fmatched, step_j)
        logger.info(f'fraction matched: {fmatched * 100}%')
        loss.backward()
        optimizer.step()
        return loss.item()



