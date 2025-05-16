import random

import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="doctawho42",
    # Set the wandb project where this run will be logged.
    project="GRAIL",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 10e-5,
        "architecture": "GRAIL",
        "dataset": "eUSPTO",
        "epochs": 10,
    },
)

# Simulate training.
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    # Log metrics to wandb.
    run.log({"acc": acc, "loss": loss})

# Finish the run and upload any remaining data.
run.finish()