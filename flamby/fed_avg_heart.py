import torch
from flamby.datasets.fed_heart_disease import FedHeartDisease, BATCH_SIZE, LR, NUM_EPOCHS_POOLED, Baseline, BaselineLoss, metric, NUM_CLIENTS
from flamby.strategies.fed_avg import FedAvg
from flamby.utils import evaluate_model_on_tests

# Load the Fed-Heart-Disease dataset
train_dataloaders_heart = [
    torch.utils.data.DataLoader(
        FedHeartDisease(center=i, train=True, pooled=False),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    for i in range(NUM_CLIENTS)
]

# Initialize the baseline models and loss functions
model_heart = Baseline()
loss_func_heart = BaselineLoss()

# Train the models using FedAvg
args_heart = {
    "training_dataloaders": train_dataloaders_heart,
    "model": model_heart,
    "loss": loss_func_heart,
    "optimizer_class": torch.optim.SGD,
    "learning_rate": LR,
    "num_updates": NUM_EPOCHS_POOLED,
    "nrounds": NUM_EPOCHS_POOLED,
}
fedavg_heart = FedAvg(**args_heart)
trained_model_heart = fedavg_heart.run()[0]

test_dataloaders_heart = [
    torch.utils.data.DataLoader(
        FedHeartDisease(center=i, train=False, pooled=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    for i in range(NUM_CLIENTS)
]

# Evaluate the trained model
results_heart = evaluate_model_on_tests(trained_model_heart, test_dataloaders_heart, metric)

print("Fed-Heart-Disease Results:")
print("---------------------------")
for center, accuracy in results_heart.items():
    print(f"Center {center}: Accuracy = {accuracy * 100:.2f}%")
print(f"\nOverall Average Accuracy: {sum(results_heart.values()) / len(results_heart) * 100:.2f}%")

