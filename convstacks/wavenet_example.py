# wavenet_example.py

from models import Wavenet
import constants
import ops
import os
import torch


def wavenet_example():

    model = Wavenet()
    checkpt_path = "./checkpoint"
    model_save_path = "./model"

    num_training_clips = 2
    data = ops.download_sample_audio(cutoff=num_training_clips)
    print(f"loaded {num_training_clips} audio samples to train on.")

    learning_rate = 1e-3
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate)
    receptive_field_size = model.receptive_field_size

    saved_epoch = 0

    print("checking for existing checkpointed model ...")
    if os.path.exists(checkpt_path):
        checkpoint = torch.load(checkpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        saved_epoch = checkpoint['epoch']
        # saved_epoch_loss = checkpoint['loss']
        print(f"loaded checkpoint model at epoch {saved_epoch}")
    else:
        print("no checkpointed model found.")

    model.train()
    print("training ...")
    for epoch in range(10**5):
        for i, audio_sample in enumerate(data):
            waveform, sample_rate, labels1, labels2, labels3 = audio_sample
            x = ops.waveform_to_input(waveform, m=constants.WaveNetConstants.AUDIO_CHANNEL_SIZE)
            cx = ops.waveform_to_categorical(waveform, m=constants.WaveNetConstants.AUDIO_CHANNEL_SIZE)
            #print(f'input shape: {x.shape}')
            y = model(x)
            #print(f'output shape: {y.shape}')

            optimizer.zero_grad()
            loss = ops.softmax_loss_fn(y, cx, receptive_field_size,
                                       show_match_fraction=True)
            loss.backward()
            optimizer.step()
            print(f'(epoch = {epoch}) loss: {loss.item()}')

            if epoch > 10**5:
                break
            if epoch % 2 == 0:
                torch.save({
                    'epoch': epoch + saved_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                }, checkpt_path
                )
                print(f"saved checkpoint at epoch {epoch + saved_epoch}")
                print(f'saving model to {model_save_path}...')
                torch.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    wavenet_example()


