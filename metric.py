import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
import cv2

import matplotlib.pyplot as plt
import numpy as np
import time
import json

from config import config

def JSONcreat(dataset_name, train_size, model_path = None, discriminator_path = None):
    information = dict()
    information['epoch'] = 0
    information['model_created'] = False
    information['model_filename'] = model_path
    information['loss'] = []
    if config['gan']:
        information['discriminator_filename'] = discriminator_path
        information['D_loss'] = []
    information['PSNR'] = []
    information['SSIM'] = []
    information['MAE'] = []
    information['train_size'] = train_size
    information['train_time'] = 0
    information['learning_rate'] = config['learning_rate']

    JSONupdate(dataset_name, information)

    return information



def JSONupdate(dataset_name, information):
    if not information['train_time'] == 0:
        for m in config['metric']:
            information[m + '_tmp'] = information[m][-1]
    JSON_dict = {
        'time': time.asctime(),
        'config': config,
        'information': information,
        'result': None
    }
    json_str = json.dumps(JSON_dict, indent=4)
    with open('train_information_'+dataset_name+'.json', 'w') as json_file:
        json_file.write(json_str)



def process_plot(plot_name, y_axis_data):
    x_axis_data = []
    for i in range(len(y_axis_data)):
        x_axis_data.append(i)

    plt.plot(x_axis_data, y_axis_data, 'bo--', alpha=0.5, linewidth=1, label='acc')

    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel(plot_name)

    plt.savefig(plot_name + '.png')



class LLIE_loss(nn.Module):
    def __init__(self):
        super(LLIE_loss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.L1 = nn.L1Loss(reduction='mean')
        self.MSE = nn.MSELoss(reduction='mean')
        self.true_labels = torch.ones(config['batch_size'], 1).float()
        if config['cuda']:
            self.true_labels = self.true_labels.cuda()
    def GAN_loss(self, discriminator_score):
        size = discriminator_score.shape[0]
        loss = self.bce(discriminator_score, self.true_labels[0:size])
        return loss

    def R_loss(self, comp, real_comp, img, target, debug):
        ssim_loss = SSIM_loss(comp[0], real_comp[0])
        L1_loss = self.L1(comp[0], real_comp[0])
        re_loss = self.L1(img, comp[0]*comp[1]) + self.L1(target, real_comp[0]*real_comp[1])
        if debug == 'Y':
            print('r_loss')
            print(ssim_loss, L1_loss, re_loss)
        return ssim_loss + L1_loss + re_loss

    # def hist_loss(self, img_batch, hist, debug):
    #     target_hist = []
    #     for i in range(config['batch_size']):
    #         # hist_r = cv2.calcHist([img_batch[i]], [0], None, [256], [0, 256])
    #         # hist_g = cv2.calcHist([img_batch[i]], [1], None, [256], [0, 256])
    #         # hist_b = cv2.calcHist([img_batch[i]], [2], None, [256], [0, 256])
    #         img = img_batch[i]*255
    #         hist_r = cv2.calcHist([img], [0], None, [256], [0, 256]) / (config['picture_size'][1]*config['picture_size'][0])
    #         hist_g = cv2.calcHist([img], [1], None, [256], [0, 256]) / (config['picture_size'][1]*config['picture_size'][0])
    #         hist_b = cv2.calcHist([img], [2], None, [256], [0, 256]) / (config['picture_size'][1]*config['picture_size'][0])
    #         target_hist.append(np.array([hist_r, hist_g, hist_b]))
    #     target_hist = torch.Tensor(np.array(target_hist))
    #     target_hist = torch.squeeze(target_hist)
    #     if config['cuda']:
    #         target_hist = target_hist.cuda()
    #     if debug == 'Y':
    #         print('hist shape')
    #         print(hist.shape, target_hist.shape, hist.max(), target_hist.max())
    #     return self.L1(hist, target_hist)*100

    def Color_loss(self, img_batch, hist, debug):
        target_hist = []
        for i in range(img_batch.shape[0]):
            img = np.around(img_batch[i]*255)
            hist_r = cv2.calcHist([img], [0], None, [256], [0, 256]) / (config['picture_size'][1]*config['picture_size'][0])
            hist_g = cv2.calcHist([img], [1], None, [256], [0, 256]) / (config['picture_size'][1]*config['picture_size'][0])
            hist_b = cv2.calcHist([img], [2], None, [256], [0, 256]) / (config['picture_size'][1]*config['picture_size'][0])
            target_hist.append(np.array([hist_r, hist_g, hist_b]))
        target_hist = torch.Tensor(np.array(target_hist))
        target_hist = torch.squeeze(target_hist)
        if config['cuda']:
            target_hist = target_hist.cuda()
        if debug == 'Y':
            print('hist shape')
            print(hist.shape, target_hist.shape, hist.max(), target_hist.max())
        return self.L1(hist, target_hist)

    def forward(self, img, target,comp, color_hist,real_comp, score, debug):
        if debug == 'Y':
            print(img.max(), target.max())
        r_loss = self.R_loss(comp, real_comp, img, target, debug)
        light_loss = self.MSE(comp[2], real_comp[1])
        # hist_loss1 = self.hist_loss(target.cpu().detach().numpy(), color_hist, debug)
        # hist_loss2 = self.hist_loss(target.cpu().detach().numpy(), real_color_hist, debug)
        # color_loss = self.L1(color_hist, real_color_hist) + hist_loss1 + hist_loss2
        # if debug == 'Y':
        #     print('color_loss')
        #     print(self.L1(color_hist, real_color_hist), hist_loss1, hist_loss2)
        color_loss = self.Color_loss(target.cpu().detach().numpy(), color_hist, debug)
        out_loss = SSIM_loss(img, target) + self.MSE(img, target)
        if debug == 'Y':
            print('out_loss')
            print(SSIM_loss(img, target), self.MSE(img, target))
        GAN_loss = self.GAN_loss(score)
        loss = config['loss_weight'][0]*r_loss + config['loss_weight'][1]*light_loss + config['loss_weight'][2]*color_loss + config['loss_weight'][3]*out_loss + config['loss_weight'][4]*GAN_loss
        return loss, np.array([loss.cpu().detach().numpy(), r_loss.cpu().detach().numpy(), light_loss.cpu().detach().numpy(), color_loss.cpu().detach().numpy(), out_loss.cpu().detach().numpy(), GAN_loss.cpu().detach().numpy()])



class discriminator_loss(nn.Module):
    def __init__(self):
        super(discriminator_loss, self).__init__()

        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.true_labels = torch.ones(config['batch_size'], 1).float()
        self.false_labels = torch.ones(config['batch_size'], 1).float()
        if config['cuda']:
            self.true_labels = self.true_labels.cuda()
            self.false_labels = self.false_labels.cuda()

    def forward(self, score_real, score_fake):
        loss = self.bce(score_real, self.true_labels[0:score_real.shape[0]]) + self.bce(score_fake, self.false_labels[0:score_fake.shape[0]])
        # return loss*size
        return loss



def SSIM_loss(img, target):
    ssim = SSIM(img, target)
    return 1-ssim



def SSIM(img, target):
    # img = torch.transpose(img, 1, 3).detach().numpy()
    # target = torch.transpose(target, 1, 3).detach().numpy()
    img = img.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    batch_ssim = 0
    for i in range(img.shape[0]):
        batch_ssim += compare_ssim(img[i], target[i], data_range=1, multichannel=True, channel_axis=0)
    return batch_ssim/img.shape[0]



def PSNR(img, target):
    img = torch.transpose(img, 1, 3).cpu().detach().numpy()
    target = torch.transpose(target, 1, 3).cpu().detach().numpy()
    # batch_psnr = 0
    # for i in range(config['batch_size']):
    #     batch_psnr += compare_psnr(target[i], img[i], data_range=1)
    return compare_psnr(target, img, data_range=1)



def eval(img, target):
    psnr = PSNR(img, target)
    ssim = SSIM(img, target)
    img = img.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    mae = np.sum(np.abs(target - img)) / np.sum(target + img)
    return psnr, ssim, mae

