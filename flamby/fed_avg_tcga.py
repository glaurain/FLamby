import torch
from flamby.datasets.fed_tcga_brca import FedTcgaBrca, BATCH_SIZE, LR, NUM_EPOCHS_POOLED, Baseline, BaselineLoss, metric, NUM_CLIENTS
from flamby.strategies.fed_avg import FedAvg
from flamby.utils import evaluate_model_on_tests

# Load the Fed-TCGA-BRCA dataset
train_dataloaders_tcga = [
    torch.utils.data.DataLoader(
        FedTcgaBrca(center=i, train=True, pooled=False),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    for i in range(NUM_CLIENTS)
]

# Initialize the baseline model and loss function for Fed-TCGA-BRCA
model_tcga = Baseline()
loss_func_tcga = BaselineLoss()

# Train the model using FedAvg for Fed-TCGA-BRCA
args_tcga = {
    "training_dataloaders": train_dataloaders_tcga,
    "model": model_tcga,
    "loss": loss_func_tcga,
    "optimizer_class": torch.optim.SGD,
    "learning_rate": LR,
    "num_updates": NUM_EPOCHS_POOLED,
    "nrounds": NUM_EPOCHS_POOLED,
}
fedavg_tcga = FedAvg(**args_tcga)
trained_model_tcga = fedavg_tcga.run()[0]

# Instantiate the test dataloaders

test_dataloaders_tcga = [
    torch.utils.data.DataLoader(
        FedTcgaBrca(center=i, train=False, pooled=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    for i in range(NUM_CLIENTS)
]

# Evaluate the trained model
results_tcga = evaluate_model_on_tests(trained_model_tcga, test_dataloaders_tcga, metric)
print("\nFed-TCGA-BRCA results:", results_tcga)

print("\nFed-TCGA-BRCA Results:")
print("---------------------------")
for center, c_index in results_tcga.items():
    print(f"Center {center}: C-Index = {c_index:.4f}")
print(f"\nOverall Average C-Index: {sum(results_tcga.values()) / len(results_tcga):.4f}")
