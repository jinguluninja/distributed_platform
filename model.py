import os
import time
import random
import numpy as np
import tensorflow as tf
import math
from tensorflow.python.ops import variable_scope as vs
from matplotlib import pyplot as plt
from nets_classification import *
import subprocess
import argparse

def get_central_files(ssh_client, central_path):
	ftp_client=ssh_client.open_sftp()
	files = ftp_client.listdir(central_path)
	for file in files:
		ftp_client.get(os.path.join(central_path, file), os.path.join('.', file))
	ftp_client.close()

def put_file(ssh_client, central_path, file):
	ftp_client=ssh_client.open_sftp()
	ftp_client.put(file, os.path.join(central_path, file))
	ftp_client.close()


class DistrSystem(object):
	def __init__(self, args, labels, ssh_client):
		self.args = args
		self.labels_dict = labels
		self.ssh_client = ssh_client

		self.images = tf.placeholder(tf.float32, shape=[None, args.img_height, args.img_width, args.img_channels])
		self.labels = tf.placeholder(tf.int32, shape=[None])
		self.loss_weights = tf.placeholder(tf.float32, shape=[None])
		self.is_training = tf.placeholder(tf.uint8)
		self.dropout = tf.placeholder(tf.float32)

		if args.model == 'custom':
			self.setup_custom_model()
		elif args.model == 'linear':
			self.setup_linear()		
		elif args.model == 'Le_Net':
			self.setup_Le_Net()				
		elif args.model == 'Alex_Net':
			self.setup_Alex_Net()						
		elif args.model == 'VGG11_Net':
			self.setup_VGG11_Net()						
		elif args.model == 'VGG13_Net':
			self.setup_VGG13_Net()						
		elif args.model == 'VGG16_Net':
			self.setup_VGG16_Net()						
		elif args.model == 'VGG19_Net':
			self.setup_VGG19_Net()						
		elif args.model == 'GoogLe_Net':
			self.setup_GoogLe_Net()						
		elif args.model == 'sparse_GoogLe_Net':
			self.setup_sparse_GoogLe_Net()						
		elif args.model == 'Inception_Net':
			self.setup_Inception_Net()						
		elif args.model == 'Res_Net':
			self.setup_Res_Net()
		else:
			raise ValueError('Invalid model.')

		self.setup_loss()   

		self.optimizer = tf.train.AdamOptimizer(self.args.lr)
		self.updates = self.optimizer.minimize(self.loss)
		self.saver = tf.train.Saver()

	def super_print(self, msg):
		print(msg)
		with open(self.args.log, 'a') as f:
			f.write(msg + '\n')

	def setup_custom_model(self):
		self.predictions = custom_model(self.images, self.is_training, self.args.num_classes, self.args.batch_size, self.dropout)
	
	def setup_linear(self):
		self.predictions = linear_model(self.images, self.is_training, self.args.img_height, self.args.img_width, self.args.img_channels, 
			self.args.num_classes, self.args.batch_size, self.dropout)

	def setup_Le_Net(self):
		self.predictions = Le_Net(self.images, self.is_training, self.args.num_classes, self.args.batch_size, self.dropout)

	def setup_Alex_Net(self):
		self.predictions = Alex_Net(self.images, self.is_training, self.args.num_classes, self.args.batch_size, self.dropout)

	def setup_VGG11_Net(self):
		self.predictions = VGG11_Net(self.images, self.is_training, self.args.num_classes, self.args.batch_size, self.dropout) 

	def setup_VGG13_Net(self):
		self.predictions = VGG13_Net(self.images, self.is_training, self.args.num_classes, self.args.batch_size, self.dropout) 

	def setup_VGG16_Net(self):
		self.predictions = VGG16_Net(self.images, self.is_training, self.args.num_classes, self.args.batch_size, self.dropout) 

	def setup_VGG19_Net(self):
		self.predictions = VGG19_Net(self.images, self.is_training, self.args.num_classes, self.args.batch_size, self.dropout) 

	def setup_GoogLe_Net(self):
		self.predictions = GoogLe_Net(self.images, self.is_training, self.args.num_classes, self.args.batch_size, self.dropout) 

	def setup_sparse_GoogLe_Net(self):
		self.predictions = sparse_GoogLe_Net(self.images, self.is_training, self.args.num_classes, self.args.batch_size, self.dropout) 

	def setup_Inception_Net(self):
		self.predictions = Inception_Net(self.images, self.is_training, self.args.num_classes, self.args.batch_size, self.dropout) 

	def setup_Res_Net(self):
		self.predictions = Res_Net(self.images, self.is_training, self.args.num_classes, self.args.batch_size, self.dropout) 

	def setup_loss(self):
		self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.predictions, weights=self.loss_weights)      

	def accuracy(self, logits, truth):
		predictions = np.argmax(logits, axis=1)
		return np.mean(0.0 + (predictions == truth))

	def initialize_model(self, session, path):
	    if path is None:
	        self.super_print('Created model with fresh parameters at Institution %s' % (self.args.inst_id))
	        session.run(tf.global_variables_initializer())
	        self.super_print('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
	    else:
	        ckpt = tf.train.get_checkpoint_state(path)
	        v2_path = ckpt.model_checkpoint_path + '.index' if ckpt else ''
	        if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
	            self.super_print('Loaded weights to institution %s' % (self.args.inst_id))
	            self.saver.restore(session, ckpt.model_checkpoint_path)


	def augment(self, data_iter):
		matrix_size = data_iter.shape[0]
		roller = np.round(float(matrix_size/7))
		ox, oy = np.random.randint(-roller, roller+1, 2)
		# do_flip = np.random.randn() > 0
		# num_rot = np.random.choice(4)
		# pow_rand = np.clip(0.05*np.random.randn(), -.2, .2) + 1.0
		add_rand = np.clip(np.random.randn() * 0.1, -.4, .4)
		# Rolling
		data_iter = np.roll(np.roll(data_iter, ox, 0), oy, 1)

		# if do_flip:
		#     data_iter = np.fliplr(data_iter)

		# data_iter = np.rot90(data_iter, num_rot)

		#data_iter = data_iter ** pow_rand

		data_iter += add_rand
		return data_iter


	def optimize(self, session, sample):
		y = [self.labels_dict[i] for i in sample]
		x = np.zeros([len(sample), self.args.img_height, self.args.img_width, self.args.img_channels], dtype=np.float32)
		for i in range(len(sample)):
			img = np.load(os.path.join(self.args.data, 'train', sample[i])).astype(np.float32)
			if self.args.augment:
				x[i] = self.augment(img)
			else:
				x[i] = img
		input_feed = {}
		input_feed[self.images] = x
		input_feed[self.labels] = y
		input_feed[self.loss_weights] = [self.loss_weights_dict[i] for i in sample]
		input_feed[self.dropout] = self.args.dropout
		input_feed[self.is_training] = 1
		output_feed = [self.updates, self.loss, self.predictions]
		outputs = session.run(output_feed, input_feed)
		return outputs[1], self.accuracy(outputs[2], np.asarray(y, dtype=int))


	def get_loss_weights(self, files_train):
		self.loss_weights_dict = {}
		if self.args.weighted_loss:  
			labels = [self.labels_dict[img] for img in files_train]
			label_num = [0 for l in range(self.args.num_classes)]
			for l in labels:
				label_num[l] += 1
			label_weights = [len(files_train)/(label_num[l]*self.args.num_classes) for l in range(self.args.num_classes)]
			for l in range(len(files_train)):
				self.loss_weights_dict[files_train[l]] = label_weights[self.labels_dict[files_train[l]]]
		else:
			self.loss_weights_dict = {img: 1.0 for img in files_train}

	def get_sample_weights(self, files_train):
		self.sample_weights_dict = {}
		if self.args.weighted_samples:            
			labels = [self.labels_dict[img] for img in files_train]
			label_num = [0 for l in range(self.args.num_classes)]
			for l in inst_labels:
				label_num[l] += 1
			sample_weights = [1./(label_num[l]*self.args.num_classes) for l in range(self.args.num_classes)]
			for l in range(len(files_train)):
				self.sample_weights_dict[files_train[l]] = sample_weights[self.labels_dict[files_train[l]]]
		else:
			for img in files_train:
				self.sample_weights_dict[img] = 1./len(files_train)

	def get_train_sample(self, files_train):
		weights = [self.sample_weights_dict[x] for x in files_train]
		sample = np.random.choice(files_train, self.args.batch_size, p=weights)
		return sample


	def train(self, session, files_train, files_val):
		self.get_loss_weights(files_train)
		self.get_sample_weights(files_train)
		n_iters_per_epoch = int(len(files_train)/self.args.batch_size)

		if self.args.inst_id == 1:
			with open('train_progress.csv', 'w') as f:
				f.write('0' + ','*(4*self.args.num_inst + 1) + '\n')
			put_file(self.ssh_client, self.args.central_path, 'train_progress.csv')
			if not os.path.exists(self.args.saved_model_name):
				os.mkdir(self.args.saved_model_name)
			if not os.path.exists('%s.tar.gz' % self.args.saved_model_name):
				subprocess.run('tar -zcvf %s.tar.gz %s' % (self.args.saved_model_name, self.args.saved_model_name), shell=True)
			self.initialize_model(session, self.args.load)

		for cycle in range(self.args.max_cycles):
			while (True):
				get_central_files(self.ssh_client, self.args.central_path)
				if not os.path.exists('train_progress.csv'):
					continue
				progress_lines = [line.strip().split(',') for line in open('train_progress.csv')]
				if len(progress_lines) == 0 or  int(progress_lines[-1][0]) != cycle:
					time.sleep(self.args.sleep_time)
					continue
				if self.args.inst_id == 1:
					if cycle == 0:
						break
					if self.args.val:
						if progress_lines[-2][-1] != '' and progress_lines[-1][1] == '':
							break
					elif progress_lines[-2][self.args.num_inst] != '' and progress_lines[-1][1] == '':
						break
				else:
					if progress_lines[-1][self.args.inst_id-1] != '' and progress_lines[-1][self.args.inst_id] == '':
						break
				time.sleep(self.args.sleep_time)
			subprocess.run('tar xvzf %s.tar.gz' % (self.args.saved_model_name), shell=True)
			self.initialize_model(session, self.args.saved_model_name)
			for i in range(n_iters_per_epoch*self.args.num_epochs):
				sample = self.get_train_sample(files_train)
				train_loss, train_acc = self.optimize(session, sample)
				self.super_print('Cycle: %s, Inst: %s, Iter: %s, train loss: %.3f, train acc: %.3f' % (cycle, self.args.inst_id, i, train_loss, train_acc))
			self.saver.save(session, os.path.join(self.args.saved_model_name, 'model.weights'))
			put_file(self.ssh_client, self.args.central_path, self.args.log)
			subprocess.run('tar -zcvf %s.tar.gz %s' % (self.args.saved_model_name, self.args.saved_model_name), shell=True)
			put_file(self.ssh_client, self.args.central_path, '%s.tar.gz' % (self.args.saved_model_name))
			progress_lines = [line.strip().split(',') for line in open('train_progress.csv')]
			progress_lines[-1][self.args.inst_id] = '1'
			with open('train_progress.csv', 'w') as f:
				for line in progress_lines:
					f.write(','.join(line) + '\n')
			put_file(self.ssh_client, self.args.central_path, 'train_progress.csv')
			if self.args.val:
				if (cycle + 1) % self.args.val_freq == 0:
					self.test(session, files_val, 'val')
			elif self.args.inst_id == self.args.num_inst:
				progress_lines.append(['' for i in range(len(progress_lines[-1]))])
				progress_lines[-1][0] = str(cycle + 1)
				with open('train_progress.csv', 'w') as f:
					for line in progress_lines:
						f.write(','.join(line) + '\n')
				put_file(self.ssh_client, self.args.central_path, 'train_progress.csv')


	def test(self, session, files_test, dataset):
		self.super_print('='*80)
		if dataset == 'val':
			while (True):
				get_central_files(self.ssh_client, self.args.central_path)
				progress_lines = [line.strip().split(',') for line in open('train_progress.csv')]
				if len(progress_lines) == 0:
					time.sleep(self.args.sleep_time)
					continue
				if progress_lines[-1][self.args.num_inst+self.args.inst_id-1] != '' and progress_lines[-1][self.args.num_inst+self.args.inst_id] == '':
					break
				time.sleep(self.args.sleep_time)
		else:
			self.initialize_model(session, self.args.load)
			while (True):
				if self.args.inst_id == 1:
					with open('test_progress.csv', 'w') as f:
						f.write(','*(3*self.args.num_inst-1) + '\n')
					put_file(self.ssh_client, self.args.central_path, 'test_progress.csv')
					break
				else:
					get_central_files(self.ssh_client, self.args.central_path)						
					if not os.path.exists('test_progress.csv'):
						time.sleep(self.args.sleep_time)
						continue
					progress_line = [line.strip().split(',') for line in open('test_progress.csv')][0]
					if progress_line[self.args.inst_id-2] != '' and progress_line[self.args.inst_id-1] == '':
						break
				time.sleep(self.args.sleep_time)			

		y = [self.labels_dict[i] for i in files_test]
		batch_indices = [0]
		index = 0
		while (True):
			index += 1
			if index == len(files_test):
				if batch_indices[-1] != index:
					batch_indices.append(index)
				break
			if index % self.args.batch_size == 0:
				batch_indices.append(index)
		num_minibatches = len(batch_indices) - 1
		losses = []
		accuracies = []
		numbers = []
		for b_end in range(1, num_minibatches + 1):
			start = batch_indices[b_end-1]
			end = batch_indices[b_end]
			files_test_batch = files_test[start:end]
			y_batch = y[start:end]  
			numbers.append(end-start)
			x = np.zeros([end-start, self.args.img_height, self.args.img_width, self.args.img_channels], dtype=np.float32)
			for i in range(len(files_test_batch)):
				img = None
				if dataset == 'train':
					img = np.load(os.path.join(self.args.data, 'train', files_test_batch[i]))
				elif dataset == 'val':
					img = np.load(os.path.join(self.args.data, 'val', files_test_batch[i]))
				else:
					img = np.load(os.path.join(self.args.data, 'test', files_test_batch[i]))
				x[i] = img.astype(np.float32)


			input_feed = {}

			input_feed[self.images] = x
			input_feed[self.labels] = y_batch
			input_feed[self.loss_weights] = [1.0 for i in files_test_batch]
			input_feed[self.dropout] = 0.0
			input_feed[self.is_training] = 0         

			output_feed = [self.loss, self.predictions]

			outputs = session.run(output_feed, input_feed)
			losses += [outputs[0]]
			accuracies += [self.accuracy(outputs[1], np.asarray(y_batch, dtype=int))]

		losses = np.asarray(losses, dtype=float)
		accuracies = np.asarray(accuracies, dtype=float)
		numbers = np.asarray(numbers, dtype=int)
		n_test = np.sum(numbers)
		acc_test = np.sum(numbers*accuracies)/n_test
		loss_test = np.sum(numbers*losses)/n_test
		if dataset == 'val':
			train_progress_lines = [line.strip().split(',') for line in open('train_progress.csv')]
			cycle = train_progress_lines[-1][0]
			train_progress_lines[-1][self.args.num_inst+self.args.inst_id] = str(n_test)
			train_progress_lines[-1][2*self.args.num_inst+self.args.inst_id] = str(loss_test)
			train_progress_lines[-1][3*self.args.num_inst+self.args.inst_id] = str(acc_test)
			self.super_print('Cycle: %s, Inst: %s, val loss: %.3f, val acc: %.3f' % (cycle, self.args.inst_id, loss_test, acc_test))
			if self.args.inst_id == self.args.num_inst:
				val_numbers = np.asarray([int(train_progress_lines[-1][i]) for i in range(self.args.num_inst+1,2*self.args.num_inst+1)], dtype=int)
				val_losses = np.asarray([float(train_progress_lines[-1][i]) for i in range(2*self.args.num_inst+1,3*self.args.num_inst+1)], dtype=float)
				val_accs = np.asarray([float(train_progress_lines[-1][i]) for i in range(3*self.args.num_inst+1,4*self.args.num_inst+1)], dtype=float)
				n_val_overall = np.sum(val_numbers)
				acc_val_overall = np.sum(val_numbers*val_accs)/n_val_overall
				loss_val_overall = np.sum(val_numbers*val_losses)/n_val_overall				
				self.super_print('='*80)
				self.super_print('Cycle: %s, combined val loss: %.3f, combined val acc: %.3f' % (cycle, loss_val_overall, acc_val_overall))
				if cycle == '0' or loss_val_overall < float(train_progress_lines[-2][-1]):
					train_progress_lines[-1][-1] = str(loss_val_overall)
					self.super_print('NEW BEST VALIDATION LOSS, SAVING BEST MODEL')
					subprocess.run('cp %s.tar.gz %s_best.tar.gz' % (self.args.saved_model_name, self.args.saved_model_name), shell=True)
					put_file(self.ssh_client, self.args.central_path, '%s_best.tar.gz' % (self.args.saved_model_name))
				else:
					train_progress_lines[-1][-1] = train_progress_lines[-2][-1]
				self.super_print('='*80)
				train_progress_lines.append(['' for i in range(len(train_progress_lines[-1]))])
				train_progress_lines[-1][0] = str(int(cycle) + 1)
			with open('train_progress.csv', 'w') as f:
				for line in train_progress_lines:
					f.write(','.join(line) + '\n')
			put_file(self.ssh_client, self.args.central_path, 'train_progress.csv')
			put_file(self.ssh_client, self.args.central_path, self.args.log)
		else:
			test_progress_lines = [line.strip().split(',') for line in open('test_progress.csv')]
			test_progress_lines[-1][self.args.inst_id-1] = str(n_test)
			test_progress_lines[-1][self.args.num_inst+self.args.inst_id-1] = str(loss_test)
			test_progress_lines[-1][2*self.args.num_inst+self.args.inst_id-1] = str(acc_test)
			self.super_print('Inst: %s, test loss: %.3f, test acc: %.3f' % (self.args.inst_id, loss_test, acc_test))
			if self.args.inst_id == self.args.num_inst:
				test_numbers = np.asarray([int(test_progress_lines[-1][i]) for i in range(0,self.args.num_inst)], dtype=int)
				test_losses = np.asarray([float(test_progress_lines[-1][i]) for i in range(self.args.num_inst,2*self.args.num_inst)], dtype=float)
				test_accs = np.asarray([float(test_progress_lines[-1][i]) for i in range(2*self.args.num_inst,3*self.args.num_inst)], dtype=float)
				n_test_overall = np.sum(test_numbers)
				acc_test_overall = np.sum(test_numbers*test_accs)/n_test_overall
				loss_test_overall = np.sum(test_numbers*test_losses)/n_test_overall				
				self.super_print('='*80)
				self.super_print('combined test loss: %.3f, combined test acc: %.3f' % (loss_test_overall, acc_test_overall))
				self.super_print('='*80)
			with open('test_progress.csv', 'w') as f:
				for line in test_progress_lines:
					f.write(','.join(line) + '\n')
			put_file(self.ssh_client, self.args.central_path, 'test_progress.csv')
			put_file(self.ssh_client, self.args.central_path, self.args.log)