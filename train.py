import logging
import utils.gpu as gpu
from model.yolov3 import Yolov3
from model.loss.yolo_loss import YoloV3Loss
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import utils.datasets as data
import time
import random
import argparse
from eval.evaluator import *
from utils.tools import *
from tensorboardX import SummaryWriter
import config.yolov3_config_voc as cfg
from utils import cosine_lr_scheduler
import visdom


# import os
# os.environ["CUDA_VISIBLE_DEVICES"]='2'

def create_vis_plot(vis, _xlabel, _ylabel, _title, _legend):
    return vis.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )

def update_vis_plot(vis, iter, loc, conf, win, update_type, epoch_size=1):
    vis.line(
        X=torch.ones((1, 3)).cpu() * iter,
        Y=torch.Tensor([loc, conf, loc+conf]).unsqueeze(0).cpu() / epoch_size,
        win=win,
        update=update_type
    )

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s - %(funcName)s - %(message)s")

class Trainer(object):
    def __init__(self,  weight_path, resume, gpu_id):
        init_seeds(0)
        self.device = gpu.select_device(gpu_id)
        self.start_epoch = 0
        self.best_mAP = 0.
        self.epochs = cfg.TRAIN["EPOCHS"]
        self.weight_path = weight_path
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        self.train_dataset = data.VocDataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=cfg.TRAIN["BATCH_SIZE"],
                                           num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                           shuffle=True)
        self.yolov3 = Yolov3().to(self.device)
        # self.yolov3.apply(tools.weights_init_normal)

        self.optimizer = optim.SGD(self.yolov3.parameters(), lr=cfg.TRAIN["LR_INIT"],
                                   momentum=cfg.TRAIN["MOMENTUM"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"])
        #self.optimizer = optim.Adam(self.yolov3.parameters(), lr = lr_init, weight_decay=0.9995)

        self.criterion = YoloV3Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                                    iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"], type=opt.loss_type)

        self.__load_model_weights(weight_path, resume)

        self.scheduler = cosine_lr_scheduler.CosineDecayLR(self.optimizer,
                                                          T_max=self.epochs*len(self.train_dataloader),
                                                          lr_init=cfg.TRAIN["LR_INIT"],
                                                          lr_min=cfg.TRAIN["LR_END"],
                                                          warmup=cfg.TRAIN["WARMUP_EPOCHS"]*len(self.train_dataloader))


    def __load_model_weights(self, weight_path, resume):
        if resume:
            last_weight = os.path.join(os.path.split(weight_path)[0], "last.pt")
            chkpt = torch.load(last_weight, map_location=self.device)
            self.yolov3.load_state_dict(chkpt['model'])

            self.start_epoch = chkpt['epoch'] + 1
            if chkpt['optimizer'] is not None:
                self.optimizer.load_state_dict(chkpt['optimizer'])
                self.best_mAP = chkpt['best_mAP']
            del chkpt
        else:
            self.yolov3.load_darknet_weights(weight_path)


    def __save_model_weights(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = os.path.join(os.path.split(self.weight_path)[0], opt.loss_type, f"best_{epoch}.pt")
        last_weight = os.path.join(os.path.split(self.weight_path)[0], opt.loss_type, f"{opt.loss_type}_{epoch}.pt")
        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': self.yolov3.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(chkpt, last_weight)

        if self.best_mAP == mAP:
            torch.save(chkpt['model'], best_weight)

        if epoch > 0 and epoch % 10 == 0:
            torch.save(chkpt, os.path.join(os.path.split(self.weight_path)[0], 'backup_epoch%g.pt'%epoch))
        del chkpt


    def train(self):
        vis = visdom.Visdom()
        vis_title = 'YOLO on ' + opt.loss_type
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot(vis, 'Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot(vis, 'Epoch', 'Loss', vis_title + " epoch loss", vis_legend)
        log = []
        print("Train datasets number is : {}".format(len(self.train_dataset)))

        iter = 0
        for epoch in range(self.start_epoch, self.epochs):
            self.yolov3.train()

            mloss = torch.zeros(4)
            for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)  in enumerate(self.train_dataloader):
                iter += 1

                self.scheduler.step(len(self.train_dataloader)*epoch + i)

                imgs = imgs.to(self.device)
                label_sbbox = label_sbbox.to(self.device)
                label_mbbox = label_mbbox.to(self.device)
                label_lbbox = label_lbbox.to(self.device)
                sbboxes = sbboxes.to(self.device)
                mbboxes = mbboxes.to(self.device)
                lbboxes = lbboxes.to(self.device)

                p, p_d = self.yolov3(imgs)

                loss, loss_giou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                  label_lbbox, sbboxes, mbboxes, lbboxes)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update running mean of tracked metrics
                loss_items = torch.tensor([loss_giou, loss_conf, loss_cls, loss])
                mloss = (mloss * i + loss_items) / (i + 1)

                # Print batch results
                if i % 500 == 0:
                    s = ('Epoch:[ %d | %d ]    Batch:[ %d | %d ]    loss_giou: %.4f    loss_conf: %.4f    loss_cls: %.4f    loss: %.4f    '
                         'lr: %g') % (epoch, self.epochs - 1, i, len(self.train_dataloader) - 1, mloss[0],mloss[1], mloss[2], mloss[3],
                                      self.optimizer.param_groups[0]['lr'])
                    logging.info(s)
                    log.append(s)
                    update_vis_plot(vis, iter, loss_giou.item(), loss_cls.item() + loss_cls.item(), iter_plot, "append")

                # multi-sclae training (320-608 pixels) every 10 batches
                if self.multi_scale_train and (i+1)%10 == 0:
                    self.train_dataset.img_size = random.choice(range(10,20)) * 32
                    #print("multi_scale_img_size : {}".format(self.train_dataset.img_size))

            update_vis_plot(vis, iter, loss_giou.item(), loss_cls.item() + loss_cls.item(), epoch_plot, "append")
            mAP = 0
            if epoch >= 40:
                print('*'*20+"Validate"+'*'*20)
                with torch.no_grad():
                    APs = Evaluator(self.yolov3, iou_threshold=0.5).APs_voc()
                    for i in APs:
                        print("{} --> mAP : {}".format(i, APs[i]))
                        log.append("{} --> mAP : {}".format(i, APs[i]))
                        mAP += APs[i]
                    mAP = mAP / self.train_dataset.num_classes
                    print('mAP:%g'%(mAP))
                    log.append(f"{epoch} Epoch mAP: {mAP}")
            if epoch >= 40 and epoch % 2 == 1:
                self.__save_model_weights(epoch, mAP)
                print('best mAP : %g' % (self.best_mAP))

        with open(f"{opt.loss_type}.txt", "w") as f:
            for i in log:
                f.write(i)


if __name__ == "__main__":

    # make random seed fixed
    random_seed = 13572220
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    print("current CPU random seed: ", torch.initial_seed())
    print("current CUDA random seed: ", torch.cuda.initial_seed())
    print(torch.randn((2,5)))

    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weight/darknet53_448.weights', help='weight file path')
    parser.add_argument('--resume', action='store_true', default=False,  help='resume training flag')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument("--loss_type", type=str, default="diou", help="determine iou loss type")
    opt = parser.parse_args()

    Trainer(weight_path=opt.weight_path,
            resume=opt.resume,
            gpu_id=opt.gpu_id).train()