import os
import random
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from datasets import PartDataset
from pointnet import PointNetCls
from utils import PointNetLoss


class Anno():
    def __init__(self):
        pass
opt = Anno()

opt.batchSize = 32
opt.num_points = 2500
opt.workers = 8
opt.nepoch = 10
opt.outf = 'cls'
opt.model = ''
opt.reg_weight = 1e-3

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = PartDataset(root='shapenetcore_partanno_segmentation_benchmark_v0', classification=True,
                      npoints=opt.num_points)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

test_dataset = PartDataset(root='shapenetcore_partanno_segmentation_benchmark_v0',
                           classification=True, train=False, npoints=opt.num_points)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

criterion = PointNetLoss(opt.reg_weight)
optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
classifier.cuda()

num_batch = len(dataset) // opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        pred, trans = classifier(points)

        cls_loss = F.nll_loss(pred, target)
        iden = torch.eye(64).unsqueeze(0).cuda()
        orth_loss = (iden - torch.bmm(trans, trans.transpose(2, 1))).norm(p=2)
        loss = cls_loss + opt.reg_weight * orth_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = pred.data.max(1)[1]
        acc = pred.eq(target.data).float().cpu().mean()
        print(f'[{epoch}: {i}/{num_batch} | train]'
              f'NLLLoss: {cls_loss.item():.4f} '
              f'OrthLoss: {orth_loss.item():.4f} '
              f'Accuracy: {acc:.4f}')

        if i % 10 == 0:
            with torch.no_grad():
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                pred, _ = classifier(points)
                loss = F.nll_loss(pred, target)
                pred = pred.data.max(1)[1]
                acc = pred.eq(target.data).float().cpu().mean()
                # print('[%d: %d/%d] %s loss: %f accuracy: %f' % (
                #     epoch, i, num_batch, blue('test'), loss.item(), acc))
                print(f'[{epoch}: {i}/{num_batch} | {blue("test")}]'
                      f'Loss: {loss.item():.4f} '
                      f'Accuracy: {acc:.4f}')

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
