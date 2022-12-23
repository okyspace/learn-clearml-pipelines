from clearml import Dataset

def create_datasets(dataset_project, dataset_name, files):
	# check if dataset exists
	# ds = Dataset.get(dataset_project=dataset_project, dataset_name=dataset_name)

	# if ds:
	# 	# exist, sync files
	# 	print('dataset exists ...')
	# 	result = ds.sync_folder(local_path=files, dataset_path=None)
	# 	print("num of files removed {} modified {} added {}".format(result[0], result[1], result[2]))
	# else:
	# create if not exists
	print('dataset does not exists ...')
	ds = Dataset.create(dataset_project=dataset_project, dataset_name=dataset_name)
	ds.add_files(path=files)

	ds.upload()
	ds.finalize()
