import pandas as pd
from CustomDataset import CustomImageDataset
from CustomCnn import CustomCnn
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch import nn

"""
Returns the patient files of this dataframe and the pathological risk label of each patient.

    :param path: path to the csv file.
    :return: a two-column dataframe.
"""


def get_patient_data(path):
    data = pd.read_csv(path)
    patient_files = data["patient_id"]
    patient_labels = data["pathological_risk_label"]
    return patient_files, patient_labels


"""
Train the model.

    :param model: model to be trained.
    :param epoch: training's epoch
    :param batch_size: batch size to use for the forward pass
    :param training_set: dataframe used as training data. Includes the
        name of the images' files and its corresponding labels.
"""


def train(model, epoch, batch_size, training_set, device):

    model.train()

    trainLoader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

    # Define the loss function and the optimizer used.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    running_loss = 0.0

    for batch_idx, batch_data in enumerate(trainLoader):

        for data in batch_data:
            data.to(device)

        images, labels = batch_data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        predictions = model(images)
        loss = criterion(predictions, labels)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch+1}, Batch: {batch_idx+1}, loss = {loss.item()}")

    print(
        f"Average loss of epoch {epoch+1}, using {batch_idx+1} batches: {running_loss / (batch_idx+1)}"
    )


"""
Validation of the model.

    :param model: model to be evaluated.
    :param epoch: training's epoch
    :param batch_size: batch size to use for the forward pass
    :param validation_set: dataframe used as validation data. Includes the
        name of the images' files and its corresponding labels.
    :return: average loss of the validation.
"""


def validation(model, epoch, batch_size, validation_set, device):

    model.eval()

    valLoader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    running_vloss = 0.0

    with torch.no_grad():
        for batch_data in valLoader:

            for data in batch_data:
                data.to(device)

            images, labels = batch_data

            # forward
            output = model(images)
            loss = criterion(output, labels)
            running_vloss += loss.item()

            # prediction
            _, predicted_labels = torch.max(output, 1)
            total += len(labels)
            correct += (predicted_labels == labels).sum().item()

        # accuracy
        accuracy = correct / total

        # average loss of the validation
        avg_loss = running_vloss / total

        print(f"Accuracy of the network on {total} images: {accuracy*100:.2f}")
        print(
            f"Average validation loss of epoch {epoch+1}, using {total} images: {avg_loss}"
        )

        return avg_loss, accuracy


"""
Save the model with the best validation score.

    :param model: model to be saved
    :param epoch: training's epoch
    :param avg_loss: average validation loss
    :param best_loss: best validation loss
    :return: the new best validation loss
"""


def model_saving(model, fold: str, epoch: int, **kwargs):
    # Track best performance, and save the model's state
    model_path = Path.cwd().joinpath("checkpoints", f"model_fold{fold}_epoch{epoch}.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "fold": fold,
            "epoch": epoch,
            **kwargs,
        },
        model_path,
    )


def main():

    # ARGPARSER WILL BE ADDED HERE.

    # Directories of (good, medium) images and (good, medium, poor) images.
    quality = ["good_medium", "good_medium_poor"]
    epochs, batch_size = 5000, 16
    classes = 2
    best_loss = 1e6
    fill_sequence = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold in Path.cwd().joinpath("data", quality[0], "folds").glob("*"):
        dataset = CustomImageDataset(fold, transform=True, common_sequences=1)
        if not fill_sequence:
            dataset = dataset.create_most_frequent_length_subset()
            dataset[0][0]
        train_dataset, val_dataset = dataset.split_data()

        # Define model.
        model = CustomCnn(classes)
        model.to(device)

        for epoch in range(epochs):

            # Training step
            train(model, epoch, batch_size, train_dataset, device)

            # Validation step
            avg_loss, accuracy = validation(
                model, epoch, batch_size, val_dataset, device
            )

            # Save the best model.
            if avg_loss < best_loss:
                best_loss = model_saving(
                    model,
                    fold.stem[-1],
                    epoch,
                    **{"avg_loss": avg_loss, "accuracy": accuracy},
                )
                best_loss = avg_loss

            print(f"Epoch {epoch} is completed")

    print("Training is completed!")


if __name__ == "__main__":
    main()
