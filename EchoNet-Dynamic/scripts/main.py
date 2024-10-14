from CustomDataset import CustomSequenceDataset
from CustomCnn import CustomCnn
from learning import *
from preprocessing import *
from pathlib import Path
import torch
import logging
import argparse
import matplotlib.pyplot as plt


def main(training, epoch_to_load):

    logging.basicConfig(
        filename="training.log" if training else "testing.log",
        format="%(asctime)s %(message)s",
        filemode="w",
        level=logging.INFO,
    )

    # Create logger object
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if training:
    
        file_list = csv_with_labels()
        split_data(file_list)

        train_data = Path.cwd() / "train_data"
        val_data = Path.cwd() / "validation_data"

        train_dataset = CustomSequenceDataset("train_files.csv", str(train_data))
        validation_dataset = CustomSequenceDataset("val_files.csv", str(val_data))

        epochs, train_batch_size, val_batch_size = 10, 32, 16
        logger.info(f"Parameters used for this experiment: epochs: {epochs}, training batch size: {train_batch_size}, validation batch size: {val_batch_size")
        
        best_loss = 1e6

        # Define model.
        model = CustomCnn()
        model.to(device)

        avg_loss_train = []
        avg_loss_val = []
        accuracy_vec = []

        for epoch in range(epochs):

            # Training step
            train_loss = train(model, epoch, train_batch_size, train_dataset, device)

            # Validation step
            val_loss, accuracy = validation(
                model, epoch, val_batch_size, validation_dataset, logger, device
            )

            avg_loss_train.append(train_loss)
            avg_loss_val.append(val_loss)

            # Save the best model.
            if val_loss < best_loss:
                best_loss = model_saving(
                    model,
                    epoch,
                    **{"val_loss": val_loss, "accuracy": accuracy},
                )
                best_loss = val_loss
                logger.info(f"Model saved. Valdation loss {val_loss}, Accuracy {accuracy}")

            print(f"Epoch {epoch+1} is completed")

        plt.figure()
        plt.plot(range(1, epochs + 1), avg_loss_train, "b-", label="Training Loss")
        plt.plot(range(1, epochs + 1), avg_loss_val, "r-", label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.savefig("learning_curve.png", dpi=300)

        plt.figure()
        plt.plot(range(1, epochs + 1), accuracy_vec)
        plt.title("Accuracy plot of validation")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy score")

        plt.savefig("accuracy_plot.png", dpi=300)

        print("Training is completed!")

    else:

        test_data = Path.cwd() / "test_data"
        test_dataset = CustomSequenceDataset("test_files.csv", str(test_data))

        model = model_loading(epoch_to_load)
        model.to(device)

        log_metrics, test_batch_size = 5, 1

        test(model, log_metrics, test_batch_size, test_dataset, logger, device)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train/Test 3D Cnn using EchoNet-Dynamic dataset')
    parser.add_argument('--training', action=argparse.BooleanOptionalAction, required=True, help='Test the network or not')
    parser.add_argument('--epoch_to_load', type=int, default=None, help="The epoch to load, mandatory if --no-training is given.")
    args = parser.parse_args()

    # If --no-training, epoch_to_load must be provided
    if not args.training and (args.epoch_to_load is None):
        parser.error("--epoch_to_load is required when --no-training is given.")
    
    main(args.training, args.epoch_to_load)