import torch
import numpy as np
from timeit import default_timer

from ufno import Net3d
from lploss import *

torch.manual_seed(0)
np.random.seed(0)


def train_UFno_sg(mode1, mode2, mode3, width):
    data_path = "/media/aita130/petrol/dy/data/MFlow/"
    train_a = torch.load(f"{data_path}dP_val_a.pt")
    train_u = torch.load(f"{data_path}dP_val_u.pt")

    print(train_a.shape)    # [b, 96, 200, 24, 12]
    print(train_u.shape)    # [b, 96, 200, 24]

    device = torch.device("cuda:0")
    model = Net3d(mode1, mode2, mode3, width).to(device)

    # prepare for calculating x direction derivatives(导数)
    time_grid = np.cumsum(np.power(1.421245, range(24)))
    time_grid /= np.max(time_grid)  # (2,)
    grid_x = train_a[0, 0, :, 0, -3]    # [200]
    grid_dx = grid_x[1:-1] + grid_x[:-2] / 2 + grid_x[2:] / 2   # [198]
    grid_dx = grid_dx[None, None, :, None].to(device)   # [1, 1, 198, 1]

    batch_size = 4
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u),
        batch_size=batch_size,
        shuffle=True,
    )

    epochs = 140
    learning_rate = 0.001
    scheduler_step = 4
    scheduler_gamma = 0.85

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )
    myloss = LpLoss(size_average=False)

    start_time = default_timer()
    train_l2 = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        counter = 0
        for x, y in train_loader:
            x = x.to(device)    # [4, 96, 200, 24, 12]
            y = y.to(device)    # [4, 96, 200, 24]
            dy = (y[:, :, 2:, :] - y[:, :, :-2, :]) / grid_dx   # [4, 96, 198, 24]

            optimizer.zero_grad()

            mask = (x[:, :, :, 0:1, 0] != 0).repeat(1, 1, 1, 24)    # [4, 96, 200, 24]
            dy = (y[:, :, 2:, :] - y[:, :, :-2, :]) / grid_dx   # [4, 96, 198, 24]

            pred = model(x).view(-1, 96, 200, 24)   # (4, 96, 200, 24)
            dy_pred = (pred[:, :, 2:, :] - pred[:, :, :-2, :]) / grid_dx    # (4, 96, 198, 24)
            ori_loss = 0
            der_loss = 0

            # original loss
            for i in range(batch_size):
                ori_loss += myloss(
                    pred[i, ...][mask[i, ...]].reshape(1, -1),
                    y[i, ...][mask[i, ...]].reshape(1, -1),
                )

            # 1st derivative loss
            dy_pred = (pred[:, :, 2:, :] - pred[:, :, :-2, :]) / grid_dx
            mask_dy = mask[:, :, :198, :]
            for i in range(batch_size):
                der_loss += myloss(
                    dy_pred[i, ...][mask_dy[i, ...]].reshape(1, -1),
                    dy[i, ...][mask_dy[i, ...]].view(1, -1),
                )

            loss = ori_loss + 0.5 * der_loss

            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

            # counter += 1
            # if counter % 100 == 0:
            #     print(
            #         f"epoch: {ep}, batch: {counter}/{len(train_loader)}, train loss: {loss.item()/batch_size:.4f}"
            #     )

        train_l2 /= train_a.shape[0]
        scheduler.step()
        t2 = default_timer()

        print(f"Epoch: {ep}, time: {(t2-t1)/60:.3f} m, train loss: {train_l2:.4f}")
        # print(f"epoch: {ep}, train loss: {train_l2/train_a.shape[0]:.4f}")

        lr_ = optimizer.param_groups[0]["lr"]
        if ep % 2 == 0:
            PATH = f"/media/aita130/petrol/dy/UFNO/saved_models/dP_UFNO_{ep}ep_{width}width_{mode1}m1_{mode2}m2_{train_a.shape[0]}train_{lr_:.2e}lr"
            torch.save(model, PATH)

    elapsed = default_timer() - start_time
    el_min = elapsed / 60
    print("Training time: %.3f" % (elapsed))
    print(f"Total training time: {el_min:.3f} min")


if __name__ == "__main__":
    mode = "train"

    mode1 = 10
    mode2 = 10
    mode3 = 10
    width = 36
    if mode == "train":
        train_UFno_sg(mode1, mode2, mode3, width)
