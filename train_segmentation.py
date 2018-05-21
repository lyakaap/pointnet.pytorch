import os
import random
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from datasets import PartDataset
from pointnet import PointNetDenseCls
from utils import PointNetLoss


class Anno():
    def __init__(self):
        pass
opt = Anno()

opt.batchSize = 32
opt.num_points = 2500
opt.workers = 8
opt.nepoch = 10
opt.outf = 'seg'
opt.model = ''
opt.reg_weight = 1e-3

print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = PartDataset(root='shapenetcore_partanno_segmentation_benchmark_v0', classification=False,
                      class_choice=['Chair'])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

test_dataset = PartDataset(root='shapenetcore_partanno_segmentation_benchmark_v0',
                           classification=False, class_choice=['Chair'], train=False)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                             shuffle=False, num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = dataset.num_seg_classes
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

classifier = PointNetDenseCls(k=num_classes)

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
        pred = pred.view(-1, num_classes)
        target = target.view(-1) - 1
        loss = criterion(pred, target, trans)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = pred.data.max(1)[1].view(-1)
        acc = pred.eq(target.data).float().cpu().mean()
        print(f'[{epoch}: {i}/{num_batch} | train]'
              f'Loss: {loss.item():.4f} '
              f'Accuracy: {acc:.4f}')

        if i % 10 == 0:
            with torch.no_grad():
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                pred, _ = classifier(points)
                pred = pred.view(-1, num_classes)
                target = target.view(-1) - 1
                loss = F.nll_loss(pred, target)
                pred = pred.data.max(1)[1].view(-1)
                acc = pred.eq(target.data).float().cpu().mean()
                print(f'[{epoch}: {i}/{num_batch} | {blue("test")}]'
                      f'Loss: {loss.item():.4f} '
                      f'Accuracy: {acc:.4f}')

    torch.save(classifier.state_dict(), '%s/seg_model_%d.pth' % (opt.outf, epoch))
