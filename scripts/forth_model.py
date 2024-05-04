import pandas as pd
from CustomDataset import CustomSequenceDataset, CustomValDataset
from CustomCnn import CustomCnn
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import confusion_matrix
import logging


def get_patient_data(path):
    """
    Returns the patient files of this dataframe and the pathological risk label of each patient.

    :param path: path to the csv file.
    :return: a two-column dataframe.
    """

    data = pd.read_csv(path)
    patient_files = data["patient_id"]
    patient_labels = data["pathological_risk_label"]
    return patient_files, patient_labels


def train(model, epoch, batch_size, training_set, device):
    """
    Train the model.

    :param model: model to be trained.
    :param epoch: training's epoch
    :param batch_size: batch size to use for the forward pass
    :param training_set: dataframe used as training data. Includes the
        name of the images' files and its corresponding labels.
    """

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


def validation(model, epoch, batch_size, validation_set, logger, device):
    """
    Validation of the model.

    :param model: valuated model.
    :param epoch: training's epoch
    :param batch_size: batch size to use for the forward pass
    :param validation_set: dataframe used as validation data.
    :param device: processing device
    :param logger: Logger object
    :return: average loss of the validation.
    """

    model.eval()

    valLoader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    labels_list, predictions_list = [], []
    running_vloss = 0.0

    with torch.no_grad():
        for batch_data in valLoader:

            for data in batch_data:
                data.to(device)

            images, label = batch_data

            # forward
            output, _ = torch.max(model(images), 0)
            loss = criterion(output, label)
            running_vloss += loss.item()

            # prediction
            _, predicted_label = torch.max(output, 0)
            labels_list = labels_list.append(labels)
            predictions_list = predictions_list.append(predicted_labels)
            total += 1
            correct += (predicted_label == label).sum().item()

        # confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels_list, predictions_list).ravel()

        # accuracy, precision, sensitivity
        accuracy = (tp + tn)/ (tn + fp + fn + tp)
        precision = tp / (tp + fp)
        sensitivity = tp / (tp + fn)

        logger.info(f"Epoch: {epoch}, accuracy: {accuracy}, precision: {precision}, sensitivity: {sensitivity}, avg_loss: {avg_loss}")

        print(f"Metrics of the network on {total} images")
        print(f"Accuracy: {accuracy*100:.2f}")
        print(f"Precision: {precision*100:.2f}")
        print(f"Sensitivity: {sensitivity*100:.2f}")
        print(
            f"Average validation loss of epoch {epoch+1}, using {total} images: {avg_loss}"
        )

        return avg_loss, accuracy


def model_saving(model, fold: str, epoch: int, **kwargs):
    """
    Save the model with the best validation score.

    :param model: model to be saved
    :param fold: training's fold
    :param epoch: training's epoch
    """

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

    # Directories of (good, medium) images and (good, medium, poor) images.
    quality = ["good_medium", "good_medium_poor"]
    epochs, batch_size = 5000, 16
    classes = 2
    best_loss = 1e6
    fill_sequence = False
    
    logging.basicConfig(filename="training.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 

    # Create logger object
    logger=logging.getLogger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold in Path.cwd().joinpath("data", quality[0], "folds").glob("*"):
    
        logger.info(f"Fold: {fold}")
        
        dataset = CustomSequenceDataset(fold, transform=True, common_sequences=1)

        # if fill_sequence == true, fill or reduce dataset's common sequences to meet the length of the most common.
        if not fill_sequence:
            dataset = dataset.create_most_frequent_length_subset()

        validation_dataset = CustomValDataset(dataset)

        # Define model.
        model = CustomCnn(classes)
        model.to(device)

        for epoch in range(epochs):

            # Training step
            train(model, epoch, batch_size, dataset.training_data, device)

            # Validation step
            avg_loss, accuracy = validation(
                model, epoch, None, validation_dataset, logger, device
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
