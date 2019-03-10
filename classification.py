import os
import sys
import numpy as np
import tensorflow as tf
import logging
from model import DistrSystem
import subprocess
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, help='path to data directory (/path/to/data_dir/')
parser.add_argument('--num_inst', type=int, help='number of participating training institutions')
parser.add_argument('--inst_id', type=int, help='order of your institution among the num_inst institutions: int in the range [1,num_inst]\
    Must be different for each training institution')
parser.add_argument('--train', default=False, action='store_true', help='include to train')
parser.add_argument('--no-train', dest='train', action='store_false', help='include to not train')
parser.add_argument('--val', default=False, action='store_true', help='include to validate during training')
parser.add_argument('--no-val', dest='val', action='store_false', help='include to not validate during training')
parser.add_argument('--test', default=False, action='store_true', help='include to test')
parser.add_argument('--no-test', dest='test', action='store_false', help='include to not tet')
parser.add_argument('--model', type=str, help='Type of model to use: one of "custom", linear", "Le_Net", "Alex_Net", "VGG11_Net", \
    "VGG13_Net", "VGG16_Net", "VGG19_Net", "GoogLe_Net", "sparse_GoogLe_Net", "Inception_Net", or "Res_Net"')

# THE FOLLOWING ARGUMENTS ARE REQUIRED ONLY IF inst_id = 1
parser.add_argument('--load', type=str, default=None, help='path to saved weights to initialize model (required for testing, if training then None to train from scratch)')
parser.add_argument('--saved_model_name', type=str, help='Will save model to save_model_name/ folder within file_repo (do not include "/" in the name')
parser.add_argument('--log', type=str, default='log.txt', help='name of log file (must be different for each institution)')
parser.add_argument('--git_credential_cache_timeout', type=int, default=999999999, help='number of seconds to cache github credentials\
    It is important that this exceeds total training time')

# FOLLOWING ARGUMENTS MUST BE THE SAME FOR ALL PARTICIPATING INSTITUTIONS DURING TRAINING
parser.add_argument('--img_height', type=int, help='image height in pixels (must be same for each image)')
parser.add_argument('--img_width', type=int, help='image width in pixels (must be same for each image)')
parser.add_argument('--img_channels', type=int, help='number of image channels (must be same for each image)')
parser.add_argument('--num_classes', type=int, default=2, help='number of label classes')

parser.add_argument('--max_cycles', type=int, default=100, help='number of cycles of epochs to train')
parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs to train at an institution each cycle')
parser.add_argument('--val_freq', type=int, default=1, help='validates every val_freq cycles')
parser.add_argument('--sleep_time', type=int, default=60, help='time in seconds to pause between github pull requests')

parser.add_argument("--batch_size", type=int, default=32, help='training batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout probability')
parser.add_argument('--augment', type=bool, default=1, help='whether or not to perform data augmentation')

# SELECT TRUE FOR ONE OF THE FOLLOWING TO ADDRESS LABEL IMBALANCE
parser.add_argument('--weighted_loss', type=bool, default=1, help='1 to weight loss function by class')
parser.add_argument('--weighted_samples', type=bool, default=0, help='1 to weight training samples by class in random batch selection')

args = parser.parse_args()
logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(args.log), logging.StreamHandler()])


def main(_):
    subprocess.run('git pull', shell=True)
    subprocess.run("git config credential.helper 'cache --timeout=%s'" % (args.git_credential_cache_timeout), shell=True)

    labels = {line.strip().split(',')[0]: int(line.strip().split(',')[1]) for line in open(os.path.join(args.data, 'labels.csv'))}

    with tf.Session() as sess:
        print(args.train, args.val, args.test)
        classifier = DistrSystem(args, labels)
        if args.train:
            subprocess.run('rm train_progress.csv', shell=True)
            files_train = os.listdir(os.path.join(args.data, 'train'))
            files_val = os.listdir(os.path.join(args.data, 'val')) if args.val else None
            classifier.train(sess, files_train, files_val) 
        if args.test:
            subprocess.run('rm test_progress.csv', shell=True)
            files_test = os.listdir(os.path.join(args.data, 'test'))
            classifier.test(sess, files_test, 'test') 

if __name__ == "__main__":
    tf.app.run()