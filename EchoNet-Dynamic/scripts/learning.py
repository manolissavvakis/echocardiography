from CustomCnn import CustomCnn
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch import nn
from sklearn.metrics import accuracy_score, recall_score, precision_score


def train(model: CustomCnn, epoch: int, batch_size: int, training_set: Dataset, device):
    """
    Train the model.

    :param model: model to be trained.
    :param epoch: training's epoch
    :param batch_size: batch size to use for the forward pass
    :param training_set: dataset used as training data. Includes the
        name of the video's file and its corresponding label.
    """

    model.train()

    trainLoader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

    # Define the loss function and the optimizer used.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    running_loss = 0.0
    total = 0

    for batch_idx, batch_data in enumerate(trainLoader):

        video, labels = batch_data

        video, labels = video.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        predictions = model(video)
        loss = criterion(predictions, labels)
        running_loss += loss.item()
        total += 1

        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch+1}, Batch: {total}, loss = {loss.item()}")

    avg_loss = running_loss / total

    print(f"Average loss of epoch {epoch+1}, using {total} batches: {avg_loss}")

    return avg_loss


def validation(
    model: CustomCnn,
    epoch: int,
    batch_size: int,
    validation_set: Dataset,
    logger,
    device,
):
    """
    Validation of the model.

    :param model: valuated model.
    :param epoch: training's epoch
    :param batch_size: batch size to use for the forward pass
    :param validation_set: dataset used as validation data.
    :param device: processing device
    :param logger: logger object
    :return: average loss and accruacy of the validation.
    """

    # Create a weighted sampler.
    target = validation_set.file_list["Label"].tolist()
    class_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1.0 / class_count
    samples_weight = np.array([weight[t] for t in target])

    sampler = WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=False
    )

    model.eval()

    valLoader = DataLoader(validation_set, batch_size=batch_size, sampler=sampler)

    criterion = nn.CrossEntropyLoss()
    labels_list, predictions_list = [], []
    running_vloss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for video, label in valLoader:

            video, label = video.to(device), label.to(device)

            # forward
            output = model(video)
            predicted_label = torch.argmax(output, dim=1)

            loss = criterion(output, label)
            running_vloss += loss.item()

            # prediction
            labels_list.extend(label.tolist())
            predictions_list.extend(predicted_label.tolist())
            total += 1
            correct += sum(predicted_label == label).item()

        # accuracy, precision, sensitivity and average validation loss
        accuracy = accuracy_score(labels_list, predictions_list)
        precision = precision_score(labels_list, predictions_list)
        sensitivity = recall_score(labels_list, predictions_list)
        avg_loss = running_vloss / float(total)

        logger.info(
            f"Epoch: {epoch}, accuracy: {accuracy}, precision: {precision}, sensitivity: {sensitivity}, avg_loss: {avg_loss}"
        )

        print(f"Metrics of the network on {total} images")
        print(f"Accuracy: {accuracy*100:.2f}")
        print(f"Precision: {precision*100:.2f}")
        print(f"Sensitivity: {sensitivity*100:.2f}")
        print(
            f"Average validation loss of epoch {epoch+1}, using {total} images: {avg_loss}"
        )

        return avg_loss, accuracy


def test(
    model: CustomCnn,
    log_metrics: int,
    batch_size: int,
    test_set: Dataset,
    logger,
    device,
):
    """
    Validation of the model.

    :param model: valuated model.
    :param batch_size: batch size to use for the forward pass
    :param test_set: dataset used as test data.
    :param device: processing device
    :param logger: logger object
    :return: average loss and accruacy of the validation.
    """

    model.eval()

    testLoader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    labels_list, predictions_list = [], []
    running_vloss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for video, label in testLoader:

            video, label = video.to(device), label.to(device)

            # forward
            output = model(video)
            predicted_label = torch.argmax(output, dim=1)

            loss = criterion(output, label)
            running_vloss += loss.item()

            # prediction
            labels_list.extend(label.tolist())
            predictions_list.extend(predicted_label.tolist())
            total += 1
            correct += sum(predicted_label == label).item()

        if total % log_metrics == 0:

            # accuracy, precision, sensitivity and average validation loss
            accuracy = accuracy_score(labels_list, predictions_list)
            precision = precision_score(labels_list, predictions_list)
            sensitivity = recall_score(labels_list, predictions_list)
            avg_loss = running_vloss / float(total)

            logger.info(
                f"Data tested: {total}, accuracy: {accuracy}, precision: {precision}, sensitivity: {sensitivity}, avg_loss: {avg_loss}"
            )

            print(f"Metrics of the network on {total} videos")
            print(f"Accuracy: {accuracy*100:.2f}")
            print(f"Precision: {precision*100:.2f}")
            print(f"Sensitivity: {sensitivity*100:.2f}")
            print(f"Average validation loss after using {total} videos: {avg_loss}")

        return avg_loss, accuracy


def model_saving(model: CustomCnn, epoch: int, **kwargs):
    """
    Save the model with the best validation score.

    :param model: model to be saved
    :param fold: training's fold
    :param epoch: training's epoch
    """

    model_path = Path.cwd().joinpath("checkpoints", f"model_epoch{epoch}.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            **kwargs,
        },
        model_path,
    )
