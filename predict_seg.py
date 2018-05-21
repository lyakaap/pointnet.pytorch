import numpy as np
import torch.utils.data
from datasets import PartDataset
from pointnet import PointNetDenseCls


model_path = 'seg/weights/seg_model_9.pth'
output_path = 'seg/output/seg_model_9'
class_choice = 'Chair'
num_points = 2500
test_dataset = PartDataset(root='shapenetcore_partanno_segmentation_benchmark_v0',
                           classification=False, class_choice=[class_choice], train=False)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                             shuffle=False, num_workers=4)

num_classes = test_dataset.num_seg_classes
classifier = PointNetDenseCls(k=num_classes)
classifier.cuda()
classifier.load_state_dict(torch.load(model_path))
classifier.eval()

preds = []
labels = []

for data in testdataloader:
    with torch.no_grad():
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        pred, _ = classifier(points)
        pred = pred.view(-1, num_classes)
        target = target.view(-1) - 1
        labels.append(target.data)
        pred = pred.data.max(1)[1].view(-1)
        preds.append(pred.data)


preds = torch.cat(preds, dim=0)
labels = torch.cat(labels, dim=0)
acc = preds.eq(labels).float().mean().cpu()
print('Test Accuracy (Classification): {acc:.4f}')

preds = preds.view(-1, num_points)
for pred in preds:
    path = output_path + '/' + class_choice + '/'
    np.save(path, pred)
