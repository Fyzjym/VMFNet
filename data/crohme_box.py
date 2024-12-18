import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class CROHMELabelGraphDataset(Dataset):

    def __init__(self, root, names, nc, image_size=(256, 256)):
        super(CROHMELabelGraphDataset, self).__init__()
        self.img_size = image_size
        self.nc = nc + 1
        self.root = root
        self.npy_dir = os.path.join(root, 'link_npy')

        self.names = names
        self.npy_paths = [os.path.join(self.npy_dir, name + '.npy') for name in self.names]

        self.npy = [np.load(npy_path, allow_pickle=True).item() for npy_path in self.npy_paths]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # image = torch.cat([image] * 3, dim=0)
        bbox = self.npy[index]['bbox']
        edge_type = self.npy[index]['edge_type']

        objs = bbox[:, 0].long()
        boxes = bbox[:, 1:]

        n = edge_type.shape[0]
        triples = []
        for row in range(n):
            for col in range(n):
                triples.append([row, edge_type[row, col], col])
        triples = torch.LongTensor(triples)
        # TODO layout gt
        return objs, boxes, triples


from model.layout import boxes_to_layout_matrix

def box2layout(boxes, img_size=(256, 256)):
    H, W = img_size
    return boxes_to_layout_matrix(boxes, H=H, W=W)


def crohme_collate_fn(batch):
    """
    Collate function to be used when wrapping a VgSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving categories for all objects
    - boxes: FloatTensor of shape (O, 4) giving boxes for all objects
    - triples: FloatTensor of shape (T, 3) giving all triples, where
      triples[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
    - obj_to_img: LongTensor of shape (O,) mapping objects to images;
      obj_to_img[i] = n means that objs[i] belongs to imgs[n]
    - triple_to_img: LongTensor of shape (T,) mapping triples to images;
      triple_to_img[t] = n means that triples[t] belongs to imgs[n].
    """
    # batch is a list, and each element is (image, objs, boxes, triples)
    all_objs, all_boxes, all_triples = [], [], []
    all_obj_to_img, all_triple_to_img = [], []
    obj_offset = 0
    for i, (objs, boxes, triples) in enumerate(batch):
        O, T = objs.size(0), triples.size(0)
        all_objs.append(objs)
        all_boxes.append(boxes)
        triples = triples.clone()
        triples[:, 0] += obj_offset
        triples[:, 2] += obj_offset
        all_triples.append(triples)

        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        all_triple_to_img.append(torch.LongTensor(T).fill_(i))
        obj_offset += O

    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_triples = torch.cat(all_triples)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)

    out = (all_objs, all_boxes, all_triples,
           all_obj_to_img, all_triple_to_img)
    return out


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    root_dir = '../../datasets/crohme2019'
    npy_dir = os.path.join(root_dir, 'link_npy')
    names = [name[:-4] for name in os.listdir(npy_dir)]
    ds = CROHMELabelGraphDataset(root_dir, names, nc=102)

    loader_kwargs = {
        'batch_size': 8,
        'num_workers': 4,
        'shuffle': True,
        'collate_fn': crohme_collate_fn,
    }
    train_loader = DataLoader(ds, **loader_kwargs)
    ds_iter = iter(train_loader)
    data = next(ds_iter)
    image = data[2]
    objs = data[3]
    boxes = data[4]
    layout = data[5]
    triples = data[6]
    objs_to_imgs = data[7]
    print(image.shape)
    print(objs.shape)
    print(boxes.shape)
    print(layout.shape)
    print(triples.shape)
    print(objs_to_imgs.shape)


