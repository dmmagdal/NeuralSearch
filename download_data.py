# download_data.py
# Simply python script that downloads tfds datasets locally to a drive.
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import tensorflow_datasets as tfds


def main():
	# Datasets.
	data = [
		"wikipedia/20201201.en", # https://www.tensorflow.org/datasets/catalog/wikipedia#wikipedia20201201en
		"wiki40b", # https://www.tensorflow.org/datasets/catalog/wiki40b
	]

	# Download location.
	location = "." # Current directory.

	# Go through each dataset and download them.
	for d in data:
		x = tfds.load(d, data_dir=location)

	# Exit the program.
	exit(0)


if __name__ == "__main__":
	main()