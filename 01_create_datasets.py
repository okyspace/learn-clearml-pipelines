from clearml import Task

from PIL import Image
from datasets import create_datasets

import argparse
import pathlib
import requests
import gzip
import struct
import numpy as np
import pandas as pd


def make_images(path_to_save, images, labels):
    '''
    Extract images from binary and organised them by class folder.
    '''
    # create folder for processed/images/train?test
    path_to_save.mkdir(parents=True, exist_ok=True)

    for (i, image), label in zip(enumerate(images), labels):
        # create class folder
        class_path = path_to_save / str(label)
        class_path.mkdir(parents=True, exist_ok=True)

        # create image and save
        filepath = class_path / '{}_{}.jpg'.format(label, i)
        Image.fromarray(image.reshape(28, 28)).save(filepath)

def binary2images(dataset_paths, kind, processed_images_path, output_format):
    print('dataset_paths {}'.format(dataset_paths))

    x_path, y_path = dataset_paths
    with gzip.open(x_path) as fx, gzip.open(y_path) as fy:
        fx.read(4)
        fy.read(4)
        N, = struct.unpack('>i', fy.read(4))
        if N != struct.unpack('>i', fx.read(4))[0]:
            raise RuntimeError('wrong pair of MNIST images and labels')
        fx.read(8)

        images = np.empty((N, 784), dtype=np.uint8)
        labels = np.empty(N, dtype=np.uint8)

        for i in range(N):
            labels[i] = ord(fy.read(1))
            for j in range(784):
                images[i, j] = ord(fx.read(1))

    if output_format == 'jpg':
        make_images(processed_images_path / kind, images, labels)

def download(urls, kind, path_to_save):
    '''
    Codes to download raw dataset and save in train and test folders respectively.
    Return path of raw train and test dataset.
    '''
    # create folder to save dataset
    path_to_save = path_to_save / kind
    path_to_save.mkdir(parents=True, exist_ok=True)
    # print('path_to_save {}'.format(path_to_save))

    # download from urls
    raw_ds_paths = []
    for url in urls:
        filepath = path_to_save / pathlib.Path(url).name
        # print('Saving files to {}'.format(filepath))
        raw_ds_paths.append(filepath)
        if not filepath.exists():
            res = requests.get(url)
            if res.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(res.content)
    return raw_ds_paths

def get_args():
    '''
    Provide website where dataset urls are found, 2 args for train and test. 
    If there are more than 1 urls for e.g. train, embed in array.
    '''
    parser = argparse.ArgumentParser(description='Download MNIST binary files')
    parser.add_argument('--project', default='Experimenting-MNIST')
    parser.add_argument('--task-name', default='Download-Datasets')

    parser.add_argument('--dataset-website', default='http://yann.lecun.com')
    parser.add_argument('--train-dataset-urls', default=[
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'])
    parser.add_argument('--test-dataset-urls', default=[
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'])
    parser.add_argument('--path', type=pathlib.Path, default='./data')
    parser.add_argument('--output-format', choices=['jpg'], default='jpg')

    parser.add_argument('--dataset-project', default='Dataset-MNIST')
    parser.add_argument('--dataset-name', default='MNIST-Dataset')
    args = parser.parse_args()	
    return args

def main():
    args = get_args()
    task = Task.init(project_name=args.project, task_name=args.task_name, output_uri=True)
    # task.execute_remotely()

    # download binary to path/raw/...
    raw_train_dataset_path = download(args.train_dataset_urls, 'train', args.path / 'raw')
    raw_test_dataset_path = download(args.test_dataset_urls, 'test', args.path / 'raw')
    
    # extract raw binary to images in path/processed/images and upload
    processed_images = args.path / 'processed' / 'images'
    binary2images(raw_train_dataset_path, 'train', processed_images, args.output_format)
    binary2images(raw_test_dataset_path, 'test', processed_images, args.output_format)
    print('Downloaded dataset ......')

    # create datasets
    create_datasets(args.dataset_project, args.dataset_name, processed_images)
    print('Datasets created ......')
    

if __name__ == '__main__':
	main()
