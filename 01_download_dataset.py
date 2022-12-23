from clearml import Task

import argparse
import pathlib
import requests

def download(urls, kind, path_to_save):
    '''
    Codes to download raw dataset and save in train and test folders respectively.
    '''
    # create folder to save dataset
    path_to_save = path_to_save / kind
    path_to_save.mkdir(parents=True, exist_ok=True)

    # download from urls
    for url in urls:
        filepath = path_to_save / pathlib.Path(url).name
        # print('Saving files to {}'.format(filepath))
        if not filepath.exists():
            res = requests.get(url)
            if res.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(res.content)
    return path_to_save

def get_args():
    '''
    Provide website where dataset urls are found, 2 args for train and test. 
    If there are more than 1 urls for e.g. train, embed in array.
    '''
    parser = argparse.ArgumentParser(description='Download MNIST binary files')
    parser.add_argument('--project', default='Experimenting_mnist')
    parser.add_argument('--task-name', default='download_dataset')

    parser.add_argument('--dataset-website', default='http://yann.lecun.com')
    parser.add_argument('--train-dataset-urls', default=[
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'])
    parser.add_argument('--test-dataset-urls', default=[
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'])
    parser.add_argument('--path-to-save', type=pathlib.Path, default='./data/raw')
    args = parser.parse_args()	
    return args

def main():
    args = get_args()
    task = Task.init(project_name=args.project, task_name=args.task_name, output_uri=True)
    # task.execute_remotely()

    # download and upload
    train_dataset = download(args.train_dataset_urls, 'train', args.path_to_save)
    task.upload_artifact('train_dataset', artifact_object=train_dataset)
    test_dataset = download(args.test_dataset_urls, 'test', args.path_to_save)
    task.upload_artifact('test_dataset', artifact_object=test_dataset)
    print('Uploading artifacts in the background.')
    print('Downloaded dataset completed ......')


if __name__ == '__main__':
	main()
