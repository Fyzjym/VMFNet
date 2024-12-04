

from model.generator import Sg2ImModel
from model.bbox_net import BBoxNet
# from model.generator_bak import Sg2ImModel

import torch

import os
from PIL import Image
from tqdm import tqdm
import json

from data.process import CROHME2Graph

import torchvision.transforms as T

import warnings
warnings.filterwarnings("ignore")

class ResultViewer(object):

    def __call__(self, checkpoint_pth, save_dir='eval_results'):
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = torch.load(checkpoint_pth, map_location='cpu')
        val_patchs = self.view_sample(checkpoint['val_samples'])
        print('num val samples: %d' % len(val_patchs))
        for i, patch in enumerate(tqdm(val_patchs)):
            img = Image.fromarray(patch.numpy())
            img.save(os.path.join(save_dir, 'val_%d.png' % i))

    def view_sample(self, samples):
        patch = []
        for sample in samples:
            gt_img = sample['gt_img']
            gt_box_gt_mask = sample['gt_box_gt_mask']
            gt_box_pred_mask = sample['gt_box_pred_mask']
            pred_box_pred_mask = sample['pred_box_pred_mask']
            n_imgs = gt_img.shape[0]
            for i in range(n_imgs):
                im1 = gt_img[i].squeeze()
                im2 = gt_box_gt_mask[i].squeeze()
                im3 = gt_box_pred_mask[i].squeeze()
                im4 = pred_box_pred_mask[i].squeeze()
                im = torch.cat([im1, im2, im3, im4], dim=1)
                im = im.permute(1, 2, 0)
                patch.append(im)
        return patch


CROHME_DIR = os.path.expanduser('/path/to/your')


class Predictor(object):

    def __init__(self, args, gen_checkpoint_path, box_checkpoint_path):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        with open(os.path.join(CROHME_DIR, 'vocab.json')) as f:
            self.vocab = json.load(f)
        gen_checkpoint = torch.load(gen_checkpoint_path, map_location=self.device)
        kwargs = gen_checkpoint['model_kwargs']
        with torch.no_grad():
            self.model = Sg2ImModel(**kwargs)
            self.model = self.model.to(self.device)

            raw_state_dict = gen_checkpoint['model_state']
            state_dict = {}
            for k, v in raw_state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                state_dict[k] = v
            self.model.load_state_dict(state_dict)
            self.model.eval()

        box_checkpoint = torch.load(box_checkpoint_path, map_location=self.device)
        with torch.no_grad():
            self.box_net = BBoxNet(self.vocab)
            self.box_net = self.box_net.to(self.device)

            raw_state_dict = box_checkpoint['generator']
            state_dict = {}
            for k, v in raw_state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                state_dict[k] = v
            self.box_net.load_state_dict(state_dict)
            self.box_net.eval()

        self.crohme2graph = CROHME2Graph(self.vocab)

        self.transform_256 = T.Compose([T.Resize((256, 256)), T.ToTensor()])

    def predict(self, lg_paths, img_paths=None, lpls_paths=None, content_paths=None):
        if isinstance(lg_paths, dict):
            # We just got a single scene graph, so promote it to a list
            lg_paths = [lg_paths]

        objs, triples, obj_to_img = [], [], []
        img_l, lpls_l, content_l = [], [], []
        obj_offset = 0
        for i, lg_path in enumerate(lg_paths):
            lg_objs, lg_triples = self.crohme2graph.convert(lg_path)
            for lg_obj in lg_objs:
                objs.append(lg_obj)
                obj_to_img.append(i)

            for s, p, o in lg_triples:
                triples.append([s + obj_offset, p, o + obj_offset])
            obj_offset += len(lg_objs)


        for i, img_path in enumerate(img_paths):

            img = Image.open(img_path).convert('RGB')
            img_256 = self.transform_256(img)


        for i, lpls_path in enumerate(lpls_paths):
            lpls = Image.open(lpls_path).convert('RGB')
            lpls_256 = self.transform_256(lpls)


        for i, content_path in enumerate(content_paths):
            content = Image.open(content_path).convert('RGB')
            content_256 = self.transform_256(content)

        # device = 'cpu'
        # device = 'cuda'


        objs = torch.tensor(objs, dtype=torch.int64, device=self.device)
        triples = torch.tensor(triples, dtype=torch.int64, device=self.device)
        obj_to_img = torch.tensor(obj_to_img, dtype=torch.int64, device=self.device)

        img_256 = img_256.unsqueeze(0).to(self.device)
        lpls_256 = lpls_256.unsqueeze(0).to(self.device)
        content_256 = content_256.unsqueeze(0).to(self.device)

        # img_256_noise = torch.randn_like(img_256).to(self.device)
        # lpls_256_noise = torch.randn_like(lpls_256).to(self.device)
        # content_256_noise = torch.randn_like(content_256).to(self.device)

        noise = gen_rand_noise(objs.size(0), 64, self.device)
        boxes = self.box_net(objs, triples, noise, obj_to_img=obj_to_img)

        # return self.model(objs, triples, obj_to_img=obj_to_img, boxes_gt=boxes, img=img_256_noise, lpls=lpls_256_noise, latexcontent=content_256_noise)

        return self.model(objs, triples, obj_to_img=obj_to_img, boxes_gt=boxes, img=img_256, lpls=lpls_256, latexcontent=content_256)

    def forward(self, lg_path, boxes_gt=None):
        # TODO: bbox pred
        print("run @ forward_lg1")
        return self.model.forward_lg1(lg_path, boxes_gt)


def view_box(boxes, img_size=(256, 256)):
    N = boxes.size(0)
    x0, y0, x1, y1 = boxes.split(1, 1)
    x0 = torch.round(x0 * img_size[1]).type(torch.long)
    y0 = torch.round(y0 * img_size[0]).type(torch.long)
    x1 = torch.round(x1 * img_size[1]).type(torch.long)
    y1 = torch.round(y1 * img_size[0]).type(torch.long)
    layout = torch.zeros(*img_size, dtype=torch.uint8)
    for i in range(N):
        layout[y0[i]: y1[i], x0[i]: x1[i]] += 75
    return layout


def gen_rand_noise(n_node, dim, device):
    noise = torch.randn(n_node, dim)
    noise = noise.to(device)
    return noise


if __name__ == '__main__':
    from tqdm import tqdm

    from train_image_generator import parser
    from data import imagenet_deprocess_batch

    torch.cuda.set_device(2)

    args = parser.parse_args()

    model_id = ['00600000']

    for iid in model_id:

        print("Syn model: {}".format(iid))

        checkpt_path = './save_model/' \
                       'layout_conv_sum2_new_step_{}_with_model.pt'.format(iid)

        p = Predictor(args=args,
                      gen_checkpoint_path=checkpt_path,
                      box_checkpoint_path='./box_net_40000.pkl')

        # lg_dir = '100k_symlg'

        save_dir = './save_test/model_{}'.format(iid)

        os.makedirs(save_dir + '_results', exist_ok=True)
        os.makedirs(save_dir + '_layouts_pred', exist_ok=True)

        lg_dir = '/lgpath'

        base_img = os.path.join(CROHME_DIR, 'Train_imgs')
        base_lpls = os.path.join(CROHME_DIR, 'Train_imgslpls')
        base_content = os.path.join(CROHME_DIR, 'Latex_content_process')

        for idx, symlg in enumerate(tqdm(os.listdir(lg_dir))):
            try:
                path_lg = os.path.join(lg_dir, symlg)
                path_f_name = symlg.replace(".lg", '.png')

                path_img = os.path.join(base_img, path_f_name)
                path_lpls = os.path.join(base_lpls, path_f_name)
                path_content = os.path.join(base_content, path_f_name)

                res = p.predict([path_lg], [path_img], [path_lpls], [path_content])

                # box_pred = view_box(res[3].detach()) # layout
                # box_pred_enh = view_box(res[-1].detach())
                imgs = imagenet_deprocess_batch(res[2].detach())
                # layouts = res[3].detach()
                # print(imgs.shape)
                # print(layouts.shape)
                im1 = imgs[0].squeeze()
                im1 = im1.permute(1, 2, 0)
                Image.fromarray(im1.numpy()).save(os.path.join(save_dir + '_results', symlg.split('.')[0] + '.png'))
                # Image.fromarray(box_pred.numpy()).save(os.path.join(save_dir + '_layouts_pred', symlg.split('.')[0] + '.png')) # layout
                # Image.fromarray(box_pred_enh.numpy()).save(os.path.join(lg_dir + '_layouts_enh', symlg.split('.')[0] + '.png'))

            except:
                pass



