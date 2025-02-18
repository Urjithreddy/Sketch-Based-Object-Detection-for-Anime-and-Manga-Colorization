import torch
from dataset import ColorizationDataset
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import os
from ignite.engine import Events, Engine
from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint
import glob

# Ensure the models directory exists
models_dir = "./models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Load the dataset
dataset_name = "MichaelP84/manga-colorization-dataset"
full_dataset = ColorizationDataset(dataset_name, split="train", img_size=256)

# Split the dataset into training and validation sets
train_indices, val_indices = train_test_split(list(range(len(full_dataset))), test_size=0.2, random_state=42)
train_ds = torch.utils.data.Subset(full_dataset, train_indices)
val_ds = torch.utils.data.Subset(full_dataset, val_indices)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

# Define the model, optimizer, and criterion
model = smp.Unet(
    encoder_name="efficientnet-b1",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3,  # Update to 3 channels for RGB output
)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = torch.nn.L1Loss()

# Define the training step
def train_step(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, y = batch
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()

# Define the validation step
def val_step(engine, batch):
    model.eval()
    with torch.no_grad():
        x, y = batch
        y_pred = model(x)
        loss = criterion(y_pred, y)
    return y_pred, y

# Create the training and validation engines
trainer = Engine(train_step)
evaluator = Engine(val_step)

# Attach metrics to the evaluator
val_metrics = {"val_loss": Loss(criterion)}
for name, metric in val_metrics.items():
    metric.attach(evaluator, name)

# Add event handlers to log training and validation loss
@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(engine):
    print(f"Training loss: {engine.state.output}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_epoch_end(engine):
    print(f"Epoch {engine.state.epoch} ended. Running validation...")
    evaluator.run(val_dl)
    metrics = evaluator.state.metrics
    print(f"Validation loss: {metrics['val_loss']}")

# Add a model checkpoint handler
checkpoint_handler = ModelCheckpoint(
    models_dir, "unet", n_saved=1, create_dir=True, require_empty=False,
    score_function=lambda engine: -engine.state.metrics['val_loss'],
    score_name="val_loss"
)
evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler, {"model": model})

# Run the training
trainer.run(train_dl, max_epochs=5)

# Verify that the model file is saved
model_path = os.path.join(models_dir, "unet_model_val_loss=*.pt")
if len(glob.glob(model_path)) > 0:
    print(f"Model saved successfully at {model_path}")
else:
    print(f"Model not found at {model_path}")