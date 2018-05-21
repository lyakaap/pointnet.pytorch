import torch.utils.data
from datasets import PartDataset
from pointnet import PointNetCls


model_path = 'cls/cls_model_9.pth'
num_points = 2500

test_dataset = PartDataset(root='shapenetcore_partanno_segmentation_benchmark_v0', train=False,
                           classification=True, npoints=num_points)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

classifier = PointNetCls(k=len(test_dataset.classes))
classifier.cuda()
classifier.load_state_dict(torch.load(model_path))
classifier.eval()

preds = []
labels = []

for i, data in enumerate(testdataloader, 0):
    with torch.no_grad():
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        pred, _ = classifier(points)
        preds.append(pred.data.max(1)[1])
        labels.append(target.data)

preds = torch.cat(preds, dim=0)
labels = torch.cat(labels, dim=0)
acc = preds.eq(labels).float().mean().cpu()

print('Test Accuracy (Classification): {acc:.4f}')
