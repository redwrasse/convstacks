# examples.py

from wavenetlike.models import build_wavenet
import wavenetlike.constants as constants
import wavenetlike.ops as ops
from wavenetlike.dataset import Dataset
import os
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.serialization as xser

import logging

logger = logging.getLogger(__name__)


def wavenet_example_tpu():

    dev = xm.xla_device()
    model = build_wavenet().to(dev)
    checkpt_path = "./checkpoint"
    model_save_path = "./model"

    dataset = Dataset("SPEECHCOMMANDS", cutoff=2)
    logger.info(f"loaded audio samples to train on.")

    learning_rate = 1e-3
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate)
    receptive_field_size = model.receptive_field_size

    saved_epoch = 0

    # logger.info("checking for existing checkpointed model ...")
    # if os.path.exists(checkpt_path):
    #
    #     checkpoint = xser.load(checkpt_path)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     saved_epoch = checkpoint['epoch']
    #     # saved_epoch_loss = checkpoint['loss']
    #     logger.info(f"loaded checkpoint model at epoch {saved_epoch}")
    # else:
    #     logger.info("no checkpointed model found.")

    model.train()
    logger.info("training ...")
    for epoch in range(10**5):
        for i, audio_sample in enumerate(dataset):
            waveform, sample_rate, labels1, labels2, labels3 = audio_sample
            x = ops.waveform_to_input(waveform, m=constants.WaveNetConstants.AUDIO_CHANNEL_SIZE).to(dev)
            cx = ops.waveform_to_categorical(waveform, m=constants.WaveNetConstants.AUDIO_CHANNEL_SIZE).to(dev)
            #logger.info(f'input shape: {x.shape}')
            y = model(x)
            #logger.info(f'output shape: {y.shape}')

            optimizer.zero_grad()
            loss = ops.softmax_loss_fn(y, cx, receptive_field_size,
                                       show_match_fraction=True)
            loss.backward()
            optimizer.step()
            logger.info(f'(epoch = {epoch}) loss: {loss.item()}')

            if epoch > 10**5:
                break
            # if epoch % 2 == 0:
            #     xm.save({
            #         'epoch': epoch + saved_epoch,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'loss': loss.item()
            #     }, checkpt_path
            #     )
            #     logger.info(f"saved checkpoint at epoch {epoch + saved_epoch}")
            #     logger.info(f'saving model to {model_save_path}...')
            #     xser.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    wavenet_example_tpu()


