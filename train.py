import time

from tqdm import tqdm
import numpy as np

import torch


from config import config
from model import LLIE_Model
from model import Discriminator
import metric
from metric import LLIE_loss, discriminator_loss

def save_model(model, dataname):
    model_filename = 'LLIE_Model_' + dataname + '.pth'
    torch.save(model, config['ckpt_path'] + model_filename)
    return model_filename



def train(dataset_name, train_information, train_loader, val_loader, debug):
    # 载入或构建模型
    if train_information['model_created']:
        # 载入模型
        model = torch.load(config['ckpt_path']+train_information['model_filename'])
        if config['gan']:
            discriminator = torch.load(config['ckpt_path']+train_information['discriminator_filename'])
    else:
        # 构建模型
        model = LLIE_Model()
        if config['gan']:
            discriminator = Discriminator()
    if config['cuda']:
        model = model.cuda()
        if config['gan']:
            discriminator = discriminator.cuda()
    start_epoch = train_information['epoch']
    optimizer = torch.optim.Adam(model.parameters(), lr=train_information['learning_rate'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1/(10**step))
    criterion = LLIE_loss()
    if config['cuda']:
        criterion = criterion.cuda()
    if config['gan']:
        D_optimizer = torch.optim.SGD(discriminator.parameters(), lr=config['D_learning_rate'])
        D_criterion = discriminator_loss()
        if config['cuda']:
            D_criterion = D_criterion.cuda()
    for epoch in range(start_epoch, config['max_epoches']):
        print('epoch: ', epoch + 1, '/', config['max_epoches'])
        print('learning rate:', optimizer.param_groups[0]['lr'])
        train_information['epoch'] = epoch
        train_information['learning_rate'] = optimizer.param_groups[0]['lr']
        loss_sum = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        if config['gan']:
            D_loss_sum = 0
        # 训练模型
        train_time = time.time()
        time.sleep(0.1)
        with tqdm(total=len(train_loader)) as pbar_train:
            pbar_train.set_description('Training...')
            for i, data in enumerate(train_loader):
                if config['cuda']:
                    img = torch.transpose(data[0], 1, 3).cuda()
                    target = torch.transpose(data[1], 1, 3).cuda()
                    ratio = torch.transpose(data[2], 1, 3).cuda()
                else:
                    img = torch.transpose(data[0], 1, 3)
                    target = torch.transpose(data[1], 1, 3)
                    ratio = torch.transpose(data[2], 1, 3)
                if debug == 'Y':
                    print('input situation:', img.max(), target.max(), ratio.max())
                if config['gan']:
                    score_real = discriminator(target)
                    _, _, enhanced_img = model(img, ratio)
                    score_fake = discriminator(enhanced_img)
                    D_loss = D_criterion(score_real, score_fake)
                    if debug == 'Y':
                        print('D_loss:', D_loss)
                    D_optimizer.zero_grad()
                    D_loss.backward()
                    D_optimizer.step()
                comp, color_hist, enhanced_img = model(img, ratio)
                real_comp = model.d_net(target)
                if config['gan']:
                    score_fake = discriminator(enhanced_img)
                    loss, loss_list = criterion(enhanced_img, target, comp, color_hist,real_comp, score_fake, debug)
                else:
                    loss, loss_list = criterion(enhanced_img, target, comp, color_hist,real_comp, None, debug)
                if debug == 'Y':
                    print('loss:', loss)
                    print(loss_list)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss_list
                if config['gan']:
                    D_loss_sum += D_loss.cpu().detach().numpy()
                pbar_train.update(1)
        if (epoch+1)%config['trans_epoch'] == 0:
            scheduler.step()
            save_model(model, dataset_name + '_' + str(epoch+1))
            if config['gan']:
                save_model(discriminator, dataset_name + '_D_' + str(epoch+1))
            metric.JSONupdate(dataset_name + str(epoch+1), train_information)
        print('epoch', epoch + 1, 'ends.')
        train_time = time.time() - train_time
        print('Evaluating...')
        psnr = 0
        ssim = 0
        mae = 0
        for data in val_loader:
            if config['cuda']:
                img = torch.transpose(data[0], 1, 3).cuda()
                target = torch.transpose(data[1], 1, 3).cuda()
                ratio = torch.transpose(data[2], 1, 3).cuda()
            else:
                img = torch.transpose(data[0], 1, 3)
                target = torch.transpose(data[1], 1, 3)
                ratio = torch.transpose(data[2], 1, 3)
            _, _, enhanced_img = model(img, ratio)
            eval_result = metric.eval(enhanced_img, target)
            psnr += eval_result[0]
            ssim += eval_result[1]
            mae += eval_result[2]
        train_information['train_time'] += train_time
        train_information['PSNR'].append(psnr / len(val_loader))
        train_information['SSIM'].append(ssim / len(val_loader))
        train_information['MAE'].append(mae / len(val_loader))
        train_information['loss'].append((loss_sum / len(train_loader)).tolist())
        if config['gan']:
            train_information['D_loss'].append(D_loss_sum / len(train_loader))
        print('model saving...')
        train_information['model_filename'] = save_model(model, dataset_name)
        if config['gan']:
            train_information['discriminator_filename'] = save_model(discriminator, dataset_name + '_D')
        train_information['model_created'] = True
        metric.JSONupdate(dataset_name, train_information)

