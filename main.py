import json
import os

import dataset
import train
import test
from config import config
import metric



if __name__ == '__main__':
    print('Welcome to LSY\'s LLIE experiment.')
    print('1: Train; 2:LOL_Train; 3: Test')
    run_mode = input('Please choose run mode: ')
    if run_mode == '1':
        print('1:Sony; 2:Fuji; 3:Double')
        dataset_id = input('Please choose dataset: ')
        dataset_name_list = ['Sony', 'Fuji', 'Double']
        dataset_name = dataset_name_list[int(dataset_id)-1]
        if dataset_id == '1' or dataset_id == '2':
            train_lines = open(config['data_path'] + dataset_name + '/' + dataset_name + '_train_list.txt', "r",
                               encoding='utf-8').readlines()
            val_lines = open(config['data_path'] + dataset_name + '/' + dataset_name + '_val_list.txt', "r",
                             encoding='utf-8').readlines()

        elif dataset_id == '3':
            train_lines_1 = open(config['data_path'] + 'Sony/Sony_train_list.txt', "r", encoding='utf-8').readlines()
            val_lines_1 = open(config['data_path'] + 'Sony/Sony_val_list.txt', "r", encoding='utf-8').readlines()
            train_lines_2 = open(config['data_path'] + 'Fuji/Fuji_train_list.txt', "r", encoding='utf-8').readlines()
            val_lines_2 = open(config['data_path'] + 'Fuji/Fuji_val_list.txt', "r", encoding='utf-8').readlines()
            train_lines = train_lines_1 + train_lines_2
            val_lines = val_lines_1 + val_lines_2
        if not os.path.exists('train_information_' + dataset_name + '.json'):
            train_information = metric.JSONcreat(dataset_name, len(train_lines))
        else:
            countinue = input('Continue? [Y/n]')
            if countinue == 'n':
                train_information = metric.JSONcreat(dataset_name, len(train_lines))
            else:
                with open('train_information_' + dataset_name + '.json') as JSONfile:
                    train_information = json.load(JSONfile)['information']
        debug = input('Debug mode? [Y/n]')
        if debug == 'Y':
            val_loader = dataset.get_dataset(dataset_name, 'val', val_lines[0:config['batch_size']*2])
            train_loader = dataset.get_dataset(dataset_name, 'train', train_lines[0:config['batch_size']*2])
        elif debug == 'n':
            val_loader = dataset.get_dataset(dataset_name, 'val', val_lines)
            train_loader = dataset.get_dataset(dataset_name, 'train', train_lines)
        train.train(dataset_name, train_information, train_loader, val_loader, debug)
    elif run_mode == '2':
        if not os.path.exists('train_information_LOL.json'):
            train_information = metric.JSONcreat('LOL', 485)
        else:
            countinue = input('Continue? [Y/n]')
            if countinue == 'n':
                train_information = metric.JSONcreat('LOL', 485)
            else:
                with open('train_information_LOL.json') as JSONfile:
                    train_information = json.load(JSONfile)['information']
        debug = input('Debug mode? [Y/n]')
        val_loader = dataset.get_dataset_LOL('val', debug)
        train_loader = dataset.get_dataset_LOL('train', debug)
        train.train('LOL', train_information, train_loader, val_loader, debug)
    elif run_mode == '3':
        print('1:Sony; 2:Fuji')
        dataset_id = input('Please choose dataset: ')
        dataset_name_list = ['Sony', 'Fuji']
        dataset_name = dataset_name_list[int(dataset_id)]
        test_lines = open(config['data_path'] + dataset_name + '/' + dataset_name + '_test_list.txt', "r",
                          encoding='utf-8').readlines()
        test.test(dataset_name, test_lines)




