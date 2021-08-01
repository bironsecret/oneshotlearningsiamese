import torch
import argparse
from signal import signal, SIGINT
from sys import exit

from torch.optim import Adam
from torch.utils import data

from core.dataset import CustomData
from core.model import OneShotModel, ContrastiveLoss

import matplotlib.pyplot as plt

import time


def handler(signal_received, frame):  # override ctrl c
    # Handle any cleanup here
    print('\nsaving..\n')
    try:
        torch.save(model, args.model_path)
    except:
        print("no model was trained. exiting..")
    exit(0)


def save():
    print("saving..")
    torch.save(model, args.model_path)
    torch.save(losses, "losses")


signal(SIGINT, handler)
parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--ds', type=str, nargs='+',
                    help='path to the dataset', default="D://projects//torchtrain//ds//train//train")
# help='path to the dataset', default="D://projects//torchtrain//ds//test//test")
parser.add_argument('--batch_size', type=int, nargs='+',
                    help='batch size', default=32)
parser.add_argument('--margin', type=float, nargs='+',
                    help='margin value', default=1.0)
parser.add_argument('--model_path', type=str, nargs='+',
                    help='path to the model', default="model")
parser.add_argument('--lr', type=float, nargs='+',
                    help='learning rate', default=0.0005)  # usual is 0.001
# parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument('--print_every', type=int, nargs='+',
                    help='print every N steps', default=15)
parser.add_argument('--epochs', type=int, nargs='+',
                    help='number of epochs', default=1000)
parser.add_argument('--cuda', type=bool, nargs='+',
                    help='use cuda', default=True)
parser.add_argument('--train_new', type=bool, nargs='+',
                    help='train new model or load old one', default=True)
parser.add_argument('--scale', type=int, nargs='+',
                    help='every Nth will not be printed, so its the more number the bigger'
                         ' graph the less data',
                    default=1)  # you can just set it same as print_every, its basically same thing
parser.add_argument('--img_size', type=int, nargs='+',
                    help="image size",
                    default=32)  # you can just set it same as print_every, its basically same thing

args = parser.parse_args()
if args.train_new:
    print("making model..")
    # model = OneShotModel()
    model = OneShotModel(hidden=args.img_size ** 2, attn_heads=8, dropout=.1, n_layers=4)
    losses = []
    num = input("how to call this one?")
    prev = 0
else:
    print("loading model and losses..")
    losses = torch.load("losses")
    try:
        model = torch.load(args.model_path)
    except:
        model = OneShotModel()
    with open("modelname", "r") as f:
        num = f.readline()
        print("using ", num)
    prev = losses[-1]  # from begin

if args.cuda:
    model = model.cuda()

dataset = CustomData(args.ds, cuda=args.cuda, size=args.img_size)
loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

criterion = ContrastiveLoss(args.margin)
with open("modelname", "w") as f:
    f.truncate(0)
    f.write(num + "\n")
    f.close()
loss_list = []
now = time.time()
print("loop start #{}..".format(num))
for ep in range(args.epochs):
    for i, (img1, img2, target) in enumerate(loader):
        now2 = time.time()
        if time.time() - now > 60 * 5:  # save every 20 minutes
            save()
            now = time.time()
        # if i + 1 > 250:
        #     break
        out1, out2 = model(img1, img2)
        loss = criterion(out1, out2, target)
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.detach().cpu().numpy())
        if i % args.print_every == 0:
            print("epoch \t {} | [{}]/[{}] \tloss: \t{} time:\t {}".format(ep + 1, i, len(loader),
                                                                           round(float(loss.data), 4), time.time()-now))
            now = time.time()

    print("_____________________________________")
    save()
    # plt.plot([losses[x] for x in range(0, len(losses), args.scale)])
    # plt.show()
    avg = sum(loss_list) / len(loss_list)
    print("average now is {}, difference {}%".format(round(avg, 5), abs(round((prev - avg), 4) * 100)))
    prev = avg

plt.plot([losses[x] for x in range(0, len(losses), args.scale)])
plt.savefig("plot_" + num.replace("\n", ""))
plt.show()
print("saved..")
