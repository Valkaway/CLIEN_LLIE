import numpy as np
from tqdm import tqdm
import torch
import json

import dataset
import metric
from config import config

def test(dataset_name, test_lines):
    with open('train_information_' + dataset_name + '.json') as JSONfile:
        JSON_dict = json.load(JSONfile)
    train_information = JSON_dict['information']
    if train_information['model_created']:
        # 载入模型
        model = torch.load(config['ckpt_path']+train_information['model_filename'])
    else:
        print('Warning: Model not created!')
        return

    # 训练过程展示
    print('training epoches:', train_information['epoch'])
    for metric_name in config['metric']:
        metric.process_plot(metric_name, train_information[metric_name])

    # 读取数据集并测试
    test_loader = dataset.get_dataset(dataset_name, 'test', test_lines)
    psnr = 0
    ssim = 0
    mae = 0
    with tqdm(total=len(test_loader)) as pbar_test:
        pbar_test.set_description('Testing...')
        for i, data in enumerate(test_loader):
            _, _, enhanced_img = model(torch.transpose(data[0], 1, 3), torch.transpose(data[2], 1, 3))
            eval_result = metric.eval(enhanced_img, torch.transpose(data[1], 1, 3))
            psnr += eval_result[0]
            ssim += eval_result[1]
            mae += eval_result[2]
            pbar_test.update(1)

    result = dict()
    result['PSNR'] = psnr / len(test_loader)
    result['SSIM'] = ssim / len(test_loader)
    result['MAE'] = mae / len(test_loader)

    JSON_dict['result'] = result
    json_str = json.dumps(JSON_dict, indent=4)
    with open('train_information_'+dataset_name+'.json', 'w') as json_file:
        json_file.write(json_str)



