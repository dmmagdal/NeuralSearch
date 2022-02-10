# finetune_bert.py
# Work through fine-tuning a BERT model using the tensorflow-models pip
# package. The pretrained BERT model this example uses is based on is
# available on Tensorflow Hub, to see how to use it refer to the Hub
# Appendix.
# Source: https://www.tensorflow.org/text/tutorials/fine_tune_bert
# Tensorflow 2.7
# Python 3.7
# Windows/MacOS/Linux


import os
import json
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks


def main():
	# Resources
	# This directory contains the configuration, vocabulary, and a
	# pre-trained checkpoint to use in this example.
	gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
	tf.io.gfile.listdir(gs_folder_bert)

	# Can get a pre-trained BERT encoder from Tensorflow Hub.
	hub_url_bert = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"

	# The data
	# For this example, use the GLUE MRPS dataset from TFDS. This
	# dataset is not set up so that it can be directly fed into the
	# BERT model, so this section also handles the necessary
	# preprocessing.

	# Get the dataset from Tensorflow Datasets
	# The Microsoft Research Paraphrase Corpus (Dolan & Brockett, 2005)
	# is a corpus of sentence pairs automatically extracted from online
	# news sources, with human annotations for whether the sentences in
	# the pair are semantically equivalent.
	# -> Number of labels: 2
	# -> Size of training dataset: 3668
	# -> Size of evaluation dataset: 408
	# -> Maximum sequence length of training and evaluation dataset: 128
	glue, info = tfds.load(
		"glue/mrpc", with_info=True, batch_size=-1
	) # It's small, load the whole dataset.

	print(list(glue.keys()))

	# The info object describes the dataset and its features.
	print(info.features)

	# The two classes are:
	print(info.features["label"].names)

	# Here is one example from the training set:
	glue_train = glue["train"]
	for key, value in glue_train.items():
		print(f"{key:9s}: {value[0].numpy()}")

	# BERT tokenizer
	# To fine tune a pre-trained model you need to be sure that you're
	# using exactly the same tokenization, vocabulary, and index
	# mapping as you used during training. The BERT tokenizer used in
	# this example is written in pure Python (it's not built out of
	# Tensorflow ops). So you can't just plug it into your model as a
	# keras.layer like you can with preprocessing.TextVectorization.
	# The following code rebuilds the tokenizer that was used by the
	# base model.
	# Set up tokenizer to generate Tensorflow dataset.
	tokenizer = bert.tokenization.FullTokenizer(
		vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
		do_lower_case=True
	)
	print("Vocab size:", len(tokenizer.vocab))

	# Tokenize a sentence:
	tokens = tokenizer.tokenize("Hello Tensorflow!")
	print(tokens)
	ids = tokenizer.convert_tokens_to_ids(tokens)
	print(ids)

	# Preprocess the data
	# The section manually preprocessed the dataset into the format
	# expected by the model. This dataset is small, so preprocessing
	# can be done quickly and easily in memory. For larger datasets
	# the tf_models library includes some tolls for preprocessing and
	# re-serializing a dataset. See Appendix: Re-encoding a large
	# dataset for details.
	# Encode the sentences
	# The model expectes its two input sentences to be concatenated
	# together. This input is expected to start with a [CLS] "This is
	# a classification problem" token, and each sentence should end
	# with a [SEP] "Separator" token.
	print(tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]']))


	# Start by encoding all the sentences while appending a [SEP]
	# token, and packing them into ragged-tensors:
	def encode_sentence(s):
		tokens = list(tokenizer.tokenize(s.numpy()))
		tokens.append('[SEP]')
		return tokenizer.convert_tokens_to_ids(tokens)


	sentence1 = tf.ragged.constant([
		encode_sentence(s) for s in glue_train["sentence1"]
	])
	sentence2 = tf.ragged.constant([
		encode_sentence(s) for s in glue_train["sentence2"]
	])
	print("Sentence1 shape: ", sentence1.shape.as_list())
	print("Sentence2 shape: ", sentence2.shape.as_list())

	# Now prepend a [CLS], token and concatenate the ragged tensors to
	# form a single input_word_ids tensor for each example.
	# RaggedTensor.to_tensor() zero pads the longest sequence.
	cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence1.shape[0]
	input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)
	# _ = plt.pcolormesh(input_word_ids.to_tensor())

	# Mask and input type
	# The model expects two additional inputs:
	# -> the input mask
	# -> the input type
	# The mask allows the model to clearly differentiate between the 
	# content and the padding. The mask has the same shape as the
	# input_word_ids, and contains a 1 anywhere the input_word_ids is
	# not padding.
	input_mask = tf.ones_like(input_word_ids).to_tensor()
	# plt.pcolormesh(input_mask)

	# The 'input type' also has same shape, but inside the non-padded
	# region, contains a 0 or a 1 indicating which sentence the token
	# is a part of.
	type_cls = tf.zeros_like(cls)
	type_s1 = tf.zeros_like(sentence1)
	type_s2 = tf.zeros_like(sentence2)
	input_type_ids = tf.concat(
		[type_cls, type_s1, type_s2], axis=-1
	).to_tensor()
	# plt.pcolormesh(input_word_ids)


	# Put it all together
	# Collect the above text parsing code into a single function, and
	# apply it to each split of the glue/mrpc dataset.
	def encode_sentence(s, tokenizer):
		tokens = list(tokenizer.tokenize(s))
		tokens.append('[SEP]')
		return tokenizer.convert_tokens_to_ids(tokens)


	def bert_encode(glue_dict, tokenizer):
		num_examples = len(glue_dict["sentence1"])

		sentence1 = tf.ragged.constant([
			encode_sentence(s, tokenizer)
			for s in np.array(glue_dict["sentence1"])
		])
		sentence2 = tf.ragged.constant([
			encode_sentence(s, tokenizer)
			for s in np.array(glue_dict["sentence2"])
		])

		cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence1.shape[0]
		input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

		input_mask = tf.ones_like(input_word_ids).to_tensor()

		type_cls = tf.zeros_like(cls)
		type_s1 = tf.zeros_like(sentence1)
		type_s2 = tf.zeros_like(sentence2)
		input_type_ids = tf.concat(
			[type_cls, type_s1, type_s2], axis=-1
		).to_tensor()

		inputs = {
			"input_word_ids": input_word_ids.to_tensor(),
			"input_mask": input_mask,
			"input_type_ids": input_type_ids
		}

		return inputs


	glue_train = bert_encode(glue["train"], tokenizer)
	glue_train_labels = glue["train"]["label"]

	glue_validation = bert_encode(glue["validation"], tokenizer)
	glue_validation_labels = glue["validation"]["label"]

	glue_test = bert_encode(glue["test"], tokenizer)
	glue_test_labels = glue["test"]["label"]

	# Each subset of the data has been converted to a dictionary of
	# features, and a set of labels. Each feature in the input
	# dictionary has the same shape, and the number of labels should
	# match:
	for key, value in glue_train.items():
		print(f"{key:15s} shape: {value.shape}")
	print(f"glue_train_labels shape: {glue_train_labels.shape}")

	# The model
	# Build the model. The first step is to download the configuration
	# for the pre-trained model.
	bert_config_file = os.path.join(gs_folder_bert, "bert_config.json")
	config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())
	bert_config = bert.configs.BertConfig.from_dict(config_dict)
	print(json.dumps(config_dict, indent=4))

	# The config defines the core BERT model, which is a Keras model to
	# predict the outputs of num_classes from the inputs with maximum
	# sequence length max_seq_length. This function returns both the
	# encoder and the classifier.
	bert_classifier, bert_encoder = bert.bert_models.classifier_model(
		bert_config, num_labels=2
	)

	# The classifier has three inputs and one output.
	# tf.keras.utils.plot_model(bert_classifier, show_shapes=True, dpi=48)

	# Run it on a test back of 10 examples from the training set. The
	# output is the logits for the two classes:
	glue_batch = {key: val[:10] for key, val in glue_train.items()}
	print(bert_classifier(glue_batch, training=True).numpy())

	# The TransformerEncoder in the center of the classifier above is
	# the bert_encoder. Inspecting the encoder, we see it is a stack of
	# Transformer layers connected to those same three inputs.
	# tf.keras.utils.plot_model(bert_encoder, show_shapes=True, dpi=48)

	# Restore the encoder weights
	# When built the encoder is randomly initialized. Restore the
	# encoder's weights from the checkpoint:
	checkpoint = tf.train.Checkpoint(encoder=bert_encoder)
	checkpoint.read(
		os.path.join(gs_folder_bert, "bert_model.ckpt")
	).assert_consumed()

	# Note: The pretrained TransformerEncoder is also available on
	# Tensorflow Hub. See the Hub Appendix for details.

	# Set up the optimizer
	# BERT adopts the Adam optimizer with weight decay (aka AdamW). It
	# also employs a learning rate schedule that firstly warms up 0 and
	# and then decays to 0.
	# Set up epochs and steps.
	epochs = 3
	batch_size = 32
	eval_batch_size = 34

	train_data_size = len(glue_train_labels)
	steps_per_epoch = int(train_data_size / batch_size)
	num_train_steps = steps_per_epoch * epochs
	warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

	# Create an optimizer with learning rate schedule.
	optimizer = nlp.optimization.create_optimizer(
		2e-5, num_train_steps=num_train_steps, 
		num_warmup_steps=warmup_steps
	)

	# This returns an AdamWeightDecay optimizer with the learning rate
	# schedule set:
	print(type(optimizer))

	# To see an example of how to customize the optimizer and its
	# schedule, see the Optimizer schedule appendix.

	# Train the model
	# The metric is accuracy and we use sparse categorical cross-
	# entropy as loss.
	metrics = [
		tf.keras.metrics.SparseCategoricalAccuracy(
			"accuracy", dtype=tf.float32
		)
	]
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

	bert_classifier.compile(
		optimizer=optimizer, loss=loss, metrics=metrics
	)

	bert_classifier.fit(
		glue_train, glue_train_labels, 
		validation_data=(glue_validation, glue_validation_labels),
		batch_size=32, epochs=epochs
	)

	# Now run the fine-tuned model on a custom example to see that it
	# works. Start by encoding some sentence pairs:
	my_examples = bert_encode(
		glue_dict = {
			"sentence1": [
				"The rain in Spain falls mainly on the plain.",
				"Look I fine tuned BERT.",
			],
			"sentence2": [
				"It mostly rains on the flat lands of Spain.",
				"Is it working? This does not match."
			]
		},
		tokenizer=tokenizer
	)

	# The model should report class 1 "match" for the first example and
	# class 0 "no-match" for the second:
	result = bert_classifier(my_examples, training=False)
	result = tf.argmax(result).numpy()
	print(result)
	print(np.array(info.features["label"].names)[result])

	# Save the model
	# Often the goal of training a model is to use it for something, so
	# export the model and then restore it to be sure that it works.
	export_dir = "./saved_model"
	tf.saved_model.save(bert_classifier, export_dir=export_dir)

	reloaded = tf.saved_model.load(export_dir)
	reloaded_result = reloaded(
		[
			my_examples["input_word_ids"],
			my_examples["input_mask"],
			my_examples["input_type_ids"]
		], 
		training=False
	)

	original_result = bert_classifier(my_examples, training=False)

	# The results are (nearly) identical:
	print(original_result.numpy())
	print()
	print(reloaded_result.numpy())

	# -----------------------------------------------------------------
	#                            APPENDIX
	# -----------------------------------------------------------------

	# Re-encoding a large dataset -------------------------------------
	# This tutorial you re-encoded the dataset in memory, for clarity.
	# This was only possible because glue/mrpc is a very small dataset.
	# To deal with larger datasets tf_models library includes some
	# tools for processing and re-encoding a dataset for efficient
	# training. The first step is to describe which features of the
	# dataset should be transformed:
	processor = nlp.data.classifier_data_lib.TfdsProcessor(
		tfds_params="dataset=glue/mrpc,text_key=sentence1,text_b_key=sentence2",
		process_text_fn=bert.tokenization.convert_to_unicode
	)

	# Then apply the transformation to generate new TFRecord files.
	# Set up output of training and evaluation Tensorflow dataset.
	train_data_output_path = "./mrpc_train.tf_record"
	eval_data_output_path = "./mrpc_eval.tf_record"

	max_seq_length = 128
	batch_size = 32
	eval_batch_size = 32

	# Generate and save training data into a tf record file.
	input_meta_data = (
		nlp.data.classifier_data_lib.generate_tf_record_from_data_file(
			processor=processor,
			data_dir=None, # It is None because the data is from tfds, not local dir.
			tokenizer=tokenizer,
			train_data_output_path=train_data_output_path,
			eval_data_output_path=eval_data_output_path,
			max_seq_length=max_seq_length
		)
	)
	
	# Finally create tf.data input pipelines from those TFRecord files:
	training_dataset = bert.run_classifier.get_dataset_fn(
		train_data_output_path,
		max_seq_length,
		batch_size,
		is_training=True,
	)()
	evaluation_dataset = bert.run_classifier.get_dataset_fn(
		eval_data_output_path,
		max_seq_length,
		eval_batch_size,
		is_training=False,
	)()

	# The resulting tf.data.Datasets return (features, labels) pairs,
	# as expected by keras.Model.fit:
	print(training_dataset.element_spec)

	# Create tf.data.Dataset for training and evaluation
	# If you need to modify the data loading here is some code to get
	# you started:
	def create_classifier_dataset(file_path, seq_length, batch_size, 
			is_training):
		# Creates input dataset from (tf)records files for train/eval.
		dataset = tf.data.TFRecordDataset(file_path)
		if is_training:
			dataset = dataset.shuffle(100)
			dataset = dataset.repeat()

		def decode_record(record):
			name_to_features = {
				"input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
				"input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
				"segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
				"label_ids": tf.io.FixedLenFeature([], tf.int64),
			}
			return tf.io.parse_single_example(record, name_to_features)

		def _select_data_from_record(record):
			x = {
				"input_word_ids": record["input_ids"],
				"input_mask": record["input_mask"],
				"input_type_ids": record["segment_ids"],
			}
			y = record["label_ids"]
			return (x, y)

		dataset = dataset.map(
			decode_record, num_parallel_calls=tf.data.AUTOTUNE
		)
		dataset = dataset.map(
			_select_data_from_record,
			num_parallel_calls=tf.data.AUTOTUNE
		)
		dataset = dataset.batch(batch_size, drop_remainder=is_training)
		dataset = dataset.prefetch(tf.data.AUTOTUNE)
		return dataset

	# Set up batch sizes.
	batch_size = 32
	eval_batch_size = 32

	# Return Tensorflow dataset.
	train_dataset = create_classifier_dataset(
		train_data_output_path,
		input_meta_data["max_seq_length"],
		batch_size,
		is_training=True
	)
	evaluation_dataset = create_classifier_dataset(
		eval_data_output_path,
		input_meta_data["max_seq_length"],
		eval_batch_size,
		is_training=False
	)
	print(training_dataset.element_spec)

	# TFModels BERT on TFHub ------------------------------------------
	# You can get the BERT model off the shelf from TFHub. It would not
	# be hard to add a classifcation head on top of this
	# hub.KerasLayer.
	# Note: 350MB download.
	hub_model_name = "bert_en_uncased_L-12_H-768_A-12"
	hub_encoder = hub.KerasLayer(
		f"https://tfhub.dev/tensorflow/{hub_model_name}/3",
		trainable=True
	)
	print(f"The Hub encoder has {len(hub_encoder.trainable_variables)} trainable variables")

	# Test run it on a batch of data:
	result = hub_encoder(
		inputs=dict(
			input_word_ids=glue_train["input_word_ids"][:10],
			input_mask=glue_train["input_mask"][:10],
			input_type_ids=glue_train["input_type_ids"][:10],
		),
		training=False,
	)
	print("Pooled output shape:", result["pooled_output"].shape)
	print("Sequence output shape:", result["sequence_output"].shape)

	# At this point it would be simple to add a classification head
	# yourself. The bert_models.classifier_model function can also
	# build a classifier onto the encoder from Tensorflow Hub:
	hub_classifier = nlp.modeling.models.BertClassifier(
		bert_encoder,
		num_classes=2,
		dropout_rate=0.1,
		initializer=tf.keras.initializers.TruncatedNormal(
			stddev=0.02
		)
	)

	# The one downside to loading this model from TFHub is that the
	# structure of internal keras layers is not restored. So it's more
	# difficult to inspect or modify the model. The BertEncoder model
	# is now a single layer.
	# tf.keras.utils.plot_model(hub_classifier, show_shapes=True, dpi=64)
	# try:
	# 	tf.keras.utils.plot_model(hub_encoder, show_shapes=True, dpi=64)
	# 	assert False
	# except Exception as e:
	# 	print(f"{type(e).__name__}: {e}")

	# Low level model building ----------------------------------------
	# If you need a more control over the construction of the model
	# it's worth noting that the classifier_model function used earlier
	# is really just a thin wrapper over the
	# nlp.modeling.networks.BertEncoder and
	# nlp.modeling.models.BertClassifier classes. Just remember that if
	# you start modifying the architecture it may not be correct or
	# possible to reload the pre-trained checkpoint so you'll need to
	# retrain from scratch.
	# Build the encoder:
	bert_encoder_config = config_dict.copy()

	# You need to rename a few fields to make this work:
	bert_encoder_config["attention_dropout_rate"] = bert_encoder_config.pop("attention_probs_dropout_prob")
	bert_encoder_config["activation"] = tf_utils.get_activation(bert_encoder_config.pop("hidden_act"))
	bert_encoder_config["dropout_rate"] = bert_encoder_config.pop("hidden_dropout_prob")
	bert_encoder_config["initializer"] = tf.keras.initializers.TruncatedNormal(
		stddev=bert_encoder_config.pop("initializer_range")
	)
	bert_encoder_config["max_sequence_length"] = bert_encoder_config.pop("max_position_embeddings")
	bert_encoder_config["num_layers"] = bert_encoder_config.pop("num_hidden_layers")
	print(bert_encoder_config)

	manual_encoder = nlp.modeling.networks.BertEncoder(**bert_encoder_config)

	# Restore the weights:
	checkpoint = tf.train.Checkpoint(encoder=manual_encoder)
	checkpoint.read(
		os.path.join(gs_folder_bert, "bert_model.ckpt")
	).assert_consumed()

	# Test run it:
	result = manual_encoder(my_examples, training=True)
	print("Sequence output shape:", result[0].shape)
	print("Pooled output shape:", result[1].shape)

	# Wrap it in a classifier:
	manual_classifier = nlp.modeling.models.BertClassifier(
		bert_encoder,
		num_classes=2,
		dropout_rate=bert_encoder_config["dropout_rate"],
		initializer=bert_encoder_config["initializer"]
	)
	manual_classifier(my_examples, training=True).numpy()

	# Optimizers and schedules ----------------------------------------
	# The optimizer used to trian the model was created using the
	# nlp.optimization.create_optimizer function:
	optimizer = nlp.optimization.create_optimizer(
		2e-5, num_train_steps=num_train_steps,
		num_warmup_steps=warmup_steps
	)

	# That high level wrapper sets up the learning rate schedules and
	# the optimizer. The base learning rate schedule used here is a
	# linear decay to zero over the training run:
	epochs = 3
	batch_size = 32
	eval_batch_size = 32

	train_data_size = len(glue_train_labels)
	steps_per_epoch = int(train_data_size / batch_size)
	num_train_steps = steps_per_epoch * epochs

	decay_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
		initial_learning_rate=2e-5,
		decay_steps=num_train_steps,
		end_learning_rate=0
	)
	# plt.plot([decay_schedule(n) for n in range(num_train_steps)])

	# This, in turn is wrapped in a WarmUp schedule that linearly
	# increases the learning rate to the target value over the first
	# 10% of training.
	warmup_steps = num_train_steps * 0.1
	warmup_schedule = nlp.optimization.WarmUp(
		initial_learning_rate=2e-5,
		decay_schedule_fn=decay_schedule,
		warmup_steps=warmup_steps
	)

	# The warmup overshoots, because it warms up to the
	# "initial_learning_rate" following the original implementation.
	# You can set "initial_learning_rate=decay_schedule(warmup_steps)"
	# if you don't like the overshoot.
	# plt.plot([warmup_schedule(n) for n in range(num_train_steps)])

	# Then create the nlp.optimization.AdamWeightDecay using that
	# schedule, configured for the BERT model:
	optimizer = nlp.optimization.AdamWeightDecay(
		learning_rate=warmup_schedule,
		weight_decay_rate=0.01,
		epsilon=1e-6,
		exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
	)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()