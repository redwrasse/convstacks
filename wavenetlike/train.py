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
                 dataset_id,
                 optimizer=torch.optim.Adam,
                 learning_rate=0.0001,
                 weight_decay=0.0,
                 epoch_save_freq=100):

        self.model = model
        self.receptive_field_size = model.receptive_field_size
        self.dataset_id = dataset_id
        self.optimizer = optimizer(
            params=self.model.parameters(),
            lr=learning_rate
        )
        self.dataloader = None
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
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

        self.use_cuda = use_cuda

    def train(self,
              batch_size=32,
              epochs=10**5):

        logger.info(
            f'training with batch size of {batch_size} for {epochs} epochs ..."')

        self.model.train()

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

        num_workers = 8 if self.use_cuda else 0
        dataset = self.dataset_id.get_dataset()
        print(
            f'setting num_workers to {num_workers} since use_cuda={self.use_cuda}')

        def collate_fn(batch):
            res = []
            mn_len = 10 ** 6
            for tens, *rest in batch:
                mn_len = min(mn_len, tens.shape[1])
            for tens, *rest in batch:
                res.append(tens[:, :mn_len])
                # tens_fixed = tens[:, :mn_len]
                # res.append((tens_fixed,) + tuple(rest))
            stacked = torch.stack(res, 0)
            return stacked

        self.dataloader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      collate_fn=collate_fn,
                                                      num_workers=num_workers,
                                                      pin_memory=False)

        j = 0
        for epoch in range(epochs):
            epoch_loss = 0.
            n_samples = 0.
            for audio_sample in iter(self.dataloader):
                loss_value = stepOp.step(self.optimizer, self.model, audio_sample, j,
                                         self.writer)
                epoch_loss += loss_value
                n_samples += 1
                j += 1
            epoch_loss /= n_samples
            logger.info(f'(epoch = {epoch}) epoch loss: {epoch_loss}')

            self.writer.add_scalar("Loss/train", epoch_loss, epoch)
            self.writer.flush()

            if epoch > epochs:
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

    def step(self, optimizer, model, audio_sample, step_j, writer):
        waveform = audio_sample
        #waveform, sample_rate, labels1, labels2, labels3 = audio_sample
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

        writer.add_scalar("Fraction matched/train", fmatched, step_j)
        writer.add_scalar("Loss/train", loss, step_j)
        writer.flush()

        logger.info(f'(j={step_j}) loss: {loss.item()} fraction matched: {fmatched * 100}%')
        loss.backward()
        optimizer.step()
        return loss.item()



