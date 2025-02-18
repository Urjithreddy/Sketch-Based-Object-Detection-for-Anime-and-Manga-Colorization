import os
import torch
import numpy as np
from model import PatchDiscriminator
from dataset import ColorizationDataset
from torchflare.experiments import ModelConfig, Experiment
import torchflare.callbacks as cbs
import segmentation_models_pytorch as smp
from skimage.color import lab2rgb
from pix2pix import Pix2PixExperiment
from torchvision.utils import save_image
import warnings
from datasets import load_dataset
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Load the dataset from Hugging Face
dataset_name = "MichaelP84/manga-colorization-dataset"
dataset = load_dataset(dataset_name)

# Split the dataset into training and validation sets (80% training, 20% validation)
train_indices, valid_indices = train_test_split(list(range(len(dataset['train']))), test_size=0.2, random_state=42)
train_dataset = dataset['train'].select(train_indices)
valid_dataset = dataset['train'].select(valid_indices)

# Define the paths for saving images
save_every = 5
save_path = "./saved_images"

# Create custom dataset class instances
train_ds = ColorizationDataset(train_dataset, img_size=256)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)

valid_ds = ColorizationDataset(valid_dataset, img_size=256)
valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=8, shuffle=True)


def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    device = L.device  # Ensure all tensors are on the same device
    L = L.to(device)  # Ensure L is on the same device
    ab = ab.to(device)  # Ensure ab is on the same device
    L = (L + 1.0) * 50.0
    ab = ab * 110.0
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    images = np.stack(rgb_imgs, axis=0)
    return torch.from_numpy(images).permute(0, 3, 1, 2).to(device)  # Move back to the original device


# Overwrite load_checkpoint callback
@cbs.on_experiment_start(order=cbs.CallbackOrder.MODEL_INIT)
def unet_load_checkpoint(experiment: "Experiment"):
    model_path = "D:/pragnya/capstone project/Manga-Colorizer-main/models/unet_model_val_loss=-0.3843.pt"
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=experiment.device)
        print(f"Checkpoint keys: {ckpt.keys()}")  # Print the keys in the checkpoint
        experiment.state.model["generator"].load_state_dict(ckpt)
        print(f"Loaded checkpoint from {model_path}")
    else:
        print(f"Checkpoint file '{model_path}' not found. Skipping checkpoint loading.")


@cbs.on_epoch_end(order=cbs.CallbackOrder.EXTERNAL)
def save_generated_images(experiment: "Experiment"):
    if experiment.current_epoch % save_every == 0:
        sample_images = next(iter(valid_dl))
        inputs = sample_images[0].to(experiment.device)

        with torch.no_grad():
            gen_images = experiment.state.model["generator"](inputs)
        gen_images = gen_images.detach().cpu()
        gen_images = lab_to_rgb(inputs[:, :1, :, :], gen_images)  # Ensure correct input to lab_to_rgb

        real_images = lab_to_rgb(inputs[:, :1, :, :], sample_images[1].to(experiment.device))  # Ensure correct device
        save_image(
            gen_images,
            os.path.join(save_path, f"fake_images_{experiment.current_epoch}.jpg"),
            nrow=2,
        )
        save_image(
            real_images,
            os.path.join(save_path, f"real_images_{experiment.current_epoch}.jpg"),
            nrow=2,
        )


class CustomModelCheckpoint(cbs.ModelCheckpoint):
    def on_epoch_end(self, experiment):
        super().on_epoch_end(experiment)
        print(f"Model saved to {self.save_dir}/{self.file_name}")


callbacks = [
    CustomModelCheckpoint(
        mode="min", monitor="train_G_loss", file_name="model.bin", save_dir="./models"
    ),
    unet_load_checkpoint,
    save_generated_images,
]

config = ModelConfig(
    nn_module={"generator": smp.Unet, "discriminator": PatchDiscriminator},
    module_params={
        "generator": {
            "encoder_name": "efficientnet-b1",
            "encoder_weights": "imagenet",  # Use pretrained weights
            "in_channels": 3,  # Ensure the generator expects 3-channel RGB input
            "classes": 3,  # Ensure the generator outputs 3-channel RGB images
            "decoder_attention_type": "scse",  # Add attention to the decoder
        },
        "discriminator": {"input_channels": 6, "n_down": 4, "num_filters": 64},  # Increase model size
    },
    criterion={"BCE": "binary_cross_entropy_with_logits", "L1_LOSS": "l1_loss"},
    optimizer={"generator": "Adam", "discriminator": "Adam"},
    optimizer_params={"generator": {"lr": 5e-4}, "discriminator": {"lr": 1e-5}},  # Adjust learning rates
)

trainer = Pix2PixExperiment(lambda_l1=100, num_epochs=2, fp16=True, seed=42, device="cuda")  # Increase the number of epochs
trainer.compile_experiment(model_config=config, callbacks=callbacks)

# Explicitly save the model after training
trainer.fit_loader(train_dl)

# Save the generator model explicitly
torch.save(trainer.state.model["generator"].state_dict(), "./models/generator_model.bin")
print("Generator model saved to ./models/generator_model.bin")

# Save the discriminator model explicitly
torch.save(trainer.state.model["discriminator"].state_dict(), "./models/discriminator_model.bin")
print("Discriminator model saved to ./models/discriminator_model.bin")



downsamppling 4