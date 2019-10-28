import numpy
import theano
import theano.tensor as tensor
from nmt import RNNsearch
from binmt import BiRNNsearch
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import tools
from layer import LayerFactory
from config import * 
from optimizer import adadelta, SGD, adam, adam_slowstart
from data import DataCollection, getbatch
from mrt_utils import getMRTBatch
from pr_utils import getPRBatch, feature_word, feature_phrase, \
					feature_length, feature_attention_coverage, feature_wordcount,feature_treememory,feature_senmemory

import cPickle
import json
import argparse
import signal
import time
import datetime
import logging
import types

# tree similarity package
from zss import simple_distance, Node
from lxml import etree
from BerkeleyInterface import *
import jieba as analysis

parser = argparse.ArgumentParser("the script for training the NMT model")
parser.add_argument('-c', '--config', help = 'path to configuration file', required = True)
parser.add_argument('--debug', action = 'store_true', help = 'set verbose level for debugging')
parser.add_argument('--map', help = 'path to the mapping file')
parser.add_argument('--save-all-models', action = 'store_true', help = 'save all intermediate models')
args = parser.parse_args()

if args.debug:
	logging.basicConfig(level = logging.DEBUG,
		                format = '[%(asctime)s %(levelname)s] %(message)s',
		                datefmt = '%d %b %H:%M:%S')
	logging.debug('training with debug info')
else:
	logging.basicConfig(level = logging.INFO,
		                format = '[%(asctime)s %(levelname)s] %(message)s',
		                datefmt = '%d %b %H:%M:%S')

if __name__ == '__main__':

	'''
	# BerkeleyParser java environment setting
	JAR_PATH = r'/home/huiquan/working/berkeleyparser-master/BerkeleyParser-1.7.jar'
	GRM_PATH = r'/home/huiquan/working/berkeleyparser-master/chn_sm5.gr'
	cp = os.environ.get("BERKELEY_PARSER_JAR", JAR_PATH)
	startup(cp)
	gr = os.environ.get("BERKELEY_PARSER_GRM", GRM_PATH)
	args = {"gr": gr}
	opts = getOpts(dictToArgs(args))
	parser = loadGrammar(opts)
	# finish environment setting
    '''
    # initialize config
	config = config()
	if args.config:
		config = update_config(config, load_config(open(args.config, 'r').read()))
	print_config(config)

	if config['MRT'] or config['PR']:
		config['batchsize'] = 1  # the mini-batch size must be 1 for MRT

	mapping = None
	if args.map:
		mapping = cPickle.load(open(args.map, 'r'))

	logging.info('STEP 2: Training')
	# prepare data
	logging.info('STEP 2.1: Loading training data')
	data = DataCollection(config)
	logging.info('Done!\n')
	if config['PR']:
		logging.info('STEP 2.1 extra: Loading features')
		fls = []
		for fl in config['features_PR']:
			fls.append(eval(fl)(config, data))  # init features
		logging.info('Done!\n')

	# build model
	logging.info('STEP 2.2: Building model')
	if config['PR']:
		model = eval(config['model'])(config, fls = fls)
	else:
		model = eval(config['model'])(config)
	model.build()
	logging.info('Done!\n')

	logging.info('STEP 2.3: Building optimizer')
	trainer = eval(config['optimizer'])(config, model.creater.params)
	update_grads, update_params = trainer.build(model.cost, model.inputs)
	logging.info('Done!\n')

	# load checkpoint
	logging.info('STEP 2.4: Loading checkpoint')
	data.load_status(config['checkpoint_status'])
	model.load(config['checkpoint_model'])
	logging.info('Done!\n')

	# train
	logging.info('STEP 2.5: Online training')
	while data.num_iter < config['max_iter']:
		try:
			st = time.time()
			data.num_iter += 1
			trainx, trainy,trainxfms, trainyfms = data.next()
			x, xmask, y, ymask,xfms,yfms = getbatch(trainx, trainy,trainxfms,trainyfms, config)   #MLE
			if 'MRT' in config and config['MRT'] is True:
				x, xmask, y, ymask, MRTLoss = getMRTBatch(x, xmask, y, ymask, config, model, data)
			if config['semi_learning']:
				xm, ym = data.next_mono()
				xm, xmask, ym, ymask = getbatch(xm, ym, config)
				x, xmask, y, ymask, valid = model.get_inputs_batch(x, y, xm, ym)
			if config['PR']:
				assert config['batchsize'] == 1
				#x, xmask, y, ymask, features, ans = getPRBatch(x, xmask, y, ymask, config, model, data, fls,parser,opt) # transfer berkeley argument
		        x, xmask, y, ymask, features, ans = getPRBatch(x, xmask, y, ymask, config, model, data, fls,xfms,yfms)
			# saving checkpoint
			if data.num_iter % config['checkpoint_freq'] == 0:
				model.save(config['checkpoint_model'], data = data, mapping = mapping)
				data.save_status(config['checkpoint_status'])

			# saving and validating intermediate models
			if config['save']:
				if data.num_iter % config['save_freq'] == 0:
					# if args.save_all:
					logging.info('Saving n intermediate model')
					model.save(config['save_path'] + '/model_iter' + str(data.num_iter) + '.npz', data=data,
							   mapping=mapping)
					logging.info('Validating the model at iteration ' + str(data.num_iter))
					output_path = config['valid_dir'] + '/iter_' + str(data.num_iter) + '.trans'
					valid_input = open(config['valid_src'], 'r')
					valid_output = open(output_path, 'w')
					line = valid_input.readline()
					valid_num = 0
					# translating
					while line != '':
						line = line.strip()
						if config['PR']:
							result = model.translate_rerank(data.toindex_source(line.split(' ')))
						else:
							result = model.translate(data.toindex_source(line.split(' ')))
						print >> valid_output, data.print_target(numpy.asarray(result))
						valid_num += 1
						if valid_num % 100 == 0:
							logging.info('%d sentences translated' % valid_num)
						line = valid_input.readline()
					valid_output.close()
					valid_refs = tools.get_ref_files(config['valid_ref'])
					# logging
					data.valid_result[data.num_iter] = 100 * tools.bleu_file(output_path, valid_refs)
					data.valid_time[data.num_iter] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
					f = open('log', 'w')
					f.write(data.print_log())
					f.close()
					data.print_valid()
					logging.info('Done!\n')
					# update the best model
					if data.last_improved(last = True) == 0:
						model.save(config['save_path'] + '/model_best.npz', data = data, mapping = mapping)
					if data.last_improved() >= config['try_iter']:
						logging.info('No improvement for %d iterations. Stop training.\n' % data.last_improved())
						break

			# updating gradients
			upst = time.time()
			if 'MRT' in config and config['MRT'] is True:
				cost, grad_norm = update_grads(x, xmask, y, ymask, MRTLoss)
			elif config['semi_learning']:
				cost, grad_norm = update_grads(x, xmask, y, ymask, y, ymask, x, xmask, valid)
			elif config['PR']:
				cost, grad_norm = update_grads(x, xmask, y, ymask, features, ans)
			else:
				cost, grad_norm = update_grads(x, xmask, y, ymask)
			# NaN processing
			if numpy.isinf(cost.mean()) or numpy.isnan(cost.mean()):
				logging.warning('There is an NaN!')
			update_params()
			ed = time.time()
			data.time += ed - st
			data.updatetime += ed - upst

			data.train_cost.append(cost.mean())
			logging.debug('iteration %d: cost = %.4f, grad_norm = %.3e,' % (data.num_iter, cost.mean(), grad_norm)+
			' iter_time =  %.3f, total_time: %s' % (ed - st, tools.print_time(data.time)))
		except KeyboardInterrupt:
			logging.info('\nStop training by keyboard interruption.')
			break

	# save checkpoint
	s = signal.signal(signal.SIGINT, signal.SIG_IGN)
	logging.info('Saving model and status\n')
	model.save(config['checkpoint_model'], data = data, mapping = mapping)
	data.save_status(config['checkpoint_status'])
	logging.info('The training is completed.\n')
	signal.signal(signal.SIGINT, s)

