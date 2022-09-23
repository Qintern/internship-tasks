import random
import numpy as np
import torch
from torch.optim import Adam
from torch.nn import functional as F
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model import UNetWithResnet50Encoder
from dataset import SoilErosionDataset
from backboned_unet import Unet

EPOCHS = 100
BATCH_SIZE = 10


def main():
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # unet = UNetWithResnet50Encoder().to(device)
    unet = Unet(backbone_name='densenet161', classes=1).to(device)

    dataset = SoilErosionDataset()
    train_size = int(len(dataset) * 0.794)
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=(-90, 90), scale=(0.85, 1.0)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std)
    ])
    transform_test = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std)
    ])

    train_set.dataset.transforms = transform_train
    test_set.dataset.transforms = transform_test

    train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=BATCH_SIZE)

    train_steps = len(train_set) // BATCH_SIZE
    test_steps = len(test_set) // BATCH_SIZE

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = Adam(unet.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.3)

    logger = {'train': [], 'test': []}
    min_test_loss = np.inf

    for epoch in range(EPOCHS):
        torch.manual_seed(1 + epoch)

        print(f"EPOCH: {epoch + 1}/{EPOCHS}")

        unet.train()
        train_loss = 0
        for (i, (x, y)) in enumerate(train_loader):
            (x, y) = (x.to(device), y.to(device))
            pred = unet(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss

        with torch.no_grad():
            unet.eval()
            test_loss = 0
            for (x, y) in test_loader:
                (x, y) = (x.to(device), y.to(device))
                pred = unet(x)
                test_loss += criterion(pred, y)

        scheduler.step()

        avg_train_loss = train_loss / train_steps
        avg_test_loss = test_loss / test_steps

        logger["train"].append(avg_train_loss.cpu().detach().numpy())
        logger["test"].append(avg_test_loss.cpu().detach().numpy())
        print(f"Average train loss: {avg_train_loss:.6f}, Average test loss: {avg_test_loss:.6f}")

        if min_test_loss > test_loss:
            print(f'Test Loss Decreased({min_test_loss:.6f}--->{test_loss:.6f}) \t Saving The Model')
            min_test_loss = test_loss
            # Saving State Dict
            info_dict = {
                'epoch': epoch,
                'net_state': unet.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }
            torch.save(info_dict, 'saved_model.pth')

    logger_df = pd.DataFrame(logger)
    logger_df.to_csv('logger.csv')


if __name__ == "__main__":
    main()
