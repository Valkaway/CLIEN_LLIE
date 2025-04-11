config = {
    'data_path': '../data/SID/',
    'ckpt_path': '../ckpt/',

    'learning_rate': 1e-4,
    'D_learning_rate': 1e-4,
    'batch_size': 16,
    'picture_size': (400, 400),
    # 'save_step': 10,
    'max_epoches': 500,
    'trans_epoch': 300,
    # 'metric': ['loss', 'D_loss', 'PSNR', 'SSIM', 'MAE'],
    'metric': ['loss', 'PSNR', 'SSIM', 'MAE'],
    'cuda': True,
    'gan': False,
    'color_loss':True,
    'r_loss':True,
    'light_loss':True,
    'loss_weight':[2, 10, 20, 1, 0.5],

    'd_hist': 256
}
