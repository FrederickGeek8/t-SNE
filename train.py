from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch
import argparse
import torch.optim as optim
import numpy as np
from tsne_module import tSNE

if __name__ == "__main__":
    momentum = 0.5
    final_momentum = 0.8
    mom_switch_iter = 250
    stop_lying_iter = 100
    early_exaggeration = 12

    parser = argparse.ArgumentParser()
    parser.add_argument("-load", "--load_path", type=str, required=True)
    parser.add_argument("-n", "--number_points", type=int, default=1797)
    parser.add_argument("-epochs", "--epochs", type=int, default=1000)
    parser.add_argument("-lr", "--learning_rate", type=float, default=200)
    parser.add_argument("-save", "--save_path", type=str, default=None)
    parser.add_argument("-tb", "--tensorboard_path", type=str, default=None)
    args = parser.parse_args()

    f = open(f"{args.load_path}", "rb")
    p = np.load(f)

    p = early_exaggeration * p
    p = torch.from_numpy(p)
    if torch.cuda.is_available():
        p = p.cuda()

    model = tSNE(args.number_points)
    opt = optim.SGD(model.parameters(),
                    lr=args.learning_rate,
                    momentum=momentum)
    writer = SummaryWriter(args.tensorboard_path)
    total = torch.arange(args.number_points)
    print(f"Training with early exaggeration for {stop_lying_iter} epochs")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        model.zero_grad()
        loss = model(p)
        loss.backward()

        opt.step()

        if epoch == mom_switch_iter:
            print("Momentum switch")
            opt.param_groups[0]['momentum'] = final_momentum

        if epoch == stop_lying_iter:
            print("Early exaggeration switch")
            p /= early_exaggeration

        if epoch % 32 == 0:
            embedding = model.embedding(total).cpu().detach().numpy()
            figure = plt.figure()

            plt.scatter(embedding[:, 0], embedding[:, 1], s=14)

            writer.add_figure('images', figure, epoch)
            writer.add_scalar('loss', loss, epoch)

    writer.close()

    if (args.save_path is not None):
        print(f"Saving model to {args.save_path}")
        torch.save(model.state_dict(), f"{args.save_path}")
    else:
        embedding = model.embedding(total).cpu().detach().numpy()

        plt.scatter(embedding[:, 0], embedding[:, 1], s=14)

        plt.show()