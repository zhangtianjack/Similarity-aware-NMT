import math
import numpy
import json
import cPickle
from mrt_utils import getRefDict, calBleu
import re
import os
from nltk.parse import stanford
import urllib2
from lxml import etree
import logging
from zss import simple_distance, Node
from BerkeleyInterface import *
import sys
from collections import Counter
reload(sys)
sys.setdefaultencoding('utf8')

def getPRBatch(x, xmask, y, ymask, config, model, data, fls,xfms,yfms):

	sampleN = config['sampleN_PR']
	myL = int(config['LenRatio_PR'] * len(y))
	samples, attn = model.sample(x.squeeze(), myL, sampleN)
	#attn.reshape(samples.shape)
	#print samples.shape
	#print 'attn:', attn.shape
	#logging.info('getBRbatch')
	# format: {sentence:features(numpy array)}
	#y_dic = getUnique(fls, samples, x, y, config, attn, model,parser,opt)  # get features for y and hypos
	y_dic = getUnique(fls, samples, x, y, config, attn, model,xfms,yfms) # 1 xtree_similarity
	Y, YM, features, ans = getYM(y_dic, y, config)
	features = numpy.array(features, dtype = 'float32')
	diffN = len(features)

	X = numpy.zeros((x.shape[0], diffN), dtype = 'int64')
	x = x + X
	X = numpy.zeros((x.shape[0], diffN), dtype = 'float32')
	xmask = xmask + X
	y = Y
	ymask = YM

	assert ans >= 0

	return x, xmask, y, ymask, features, ans

#def getUnique(fls, samples, x, y, config, attn, model,parser,opt):
def getUnique(fls, samples, x, y, config, attn, model,xfms,yfms):
	dic = {}
	xn = x
	yn = y
	y = list(y.flatten())
	x = list(x.flatten())
	features = []

	# calculate feature for gold translation
	for fl in fls:
		'''
		if isinstance(fl, featureListAttn):
			attn_ans = model.get_attention(xn, numpy.ones(xn.shape, dtype = numpy.float32), yn, numpy.ones(yn.shape, dtype = numpy.float32))[0]
			add_info = [numpy.reshape(attn_ans, (attn_ans.shape[0], attn_ans.shape[1]))]
		else:
			add_info = None
		'''
		features.append(fl.getFeatures(x, cutSen(y, config), xfms,yfms,add_info = 0))
	    #features.append(fl.getFeatures(x, cutSen(y, config), add_info=add_info,parser,opt))
	dic[json.dumps(cutSen(y, config))] = numpy.concatenate(features)
	
	# calculate features for samples
	for i in range(samples.shape[0]):
		tmp = list(samples[i])
		features = []
		for fl in fls:
			'''
			if isinstance(fl, featureListAttn):
				add_info = [attn[i]]
			else:
				add_info = None
			'''
			features.append(fl.getFeatures(x, cutSen(tmp, config), xfms,yfms,add_info=1))
			#features.append(fl.getFeatures(x, cutSen(tmp, config), add_info = 1,parser,opt))  # tree only calculate once!,so the add_info have information
		dic[json.dumps(cutSen(tmp, config))] = numpy.concatenate(features)
	#for fl in fls:
	#fl.index+=1
	return dic

def getYM(y_dic,truey,config):
	ans = -1
	y = [json.loads(i) for i in y_dic]
	truey = list(truey.flatten())
	n = len(y_dic)
	features = []
	max = 0 
	idx = 0
	# find the longest sentence and the index of gold translation
	for key in y_dic:
		tmp = json.loads(key)
		tmplen = len(tmp)
		if max < tmplen:
			max = tmplen
		if truey == tmp:
			ans = idx
		idx += 1
	Y = numpy.ones((max,n), dtype = 'int64') * config['index_eos_trg']
	Ymask = numpy.zeros((max, n), dtype = 'float32')
	i = 0
	for key in y_dic:
		features.append(y_dic[key])
		tmp = json.loads(key)
		ly = len(tmp)
		Y[0:ly,i] = numpy.asarray(tmp, dtype = 'int64')
		Ymask[0:ly, i] = 1
		i += 1
	return Y, Ymask, numpy.asarray(features, dtype = 'float32'), numpy.asarray(ans, dtype = 'int64')

def my_log(a):
	if a == 0:
		return -1000000
	return math.log(a)

def cutSen(x, config):
	if config['index_eos_trg'] not in x:
		return x
	else:
		return x[:x.index(config['index_eos_trg']) + 1]

class featureList(object):

	def __init__(self):
		pass

	def getScore(self, source, hypo, add_info = None):
		return (self.feature_weight * self.getFeatures(source, hypo, add_info)).sum()

class featureListRef(featureList):

	def __init__(self):
		pass

class featureListAttn(featureList):

	def __init__(self):
		pass

class feature_word(featureList):

	def __init__(self, config, data):
		self.config = config
		self.data = data

		#load word table
		self.word_idx = {}
		self.word_s2t = []
		num_words = 0
		word_table = cPickle.load(open(config['word_table'], 'r'))
		writefile = open('word.txt', 'w')
		print 'total', len(word_table) ,'word entries'
		for i in word_table:
			if data.ivocab_src.has_key(i[0]) and data.ivocab_trg.has_key(i[1]):
				if self.word_idx.has_key(data.ivocab_src[i[0]]):
					self.word_idx[data.ivocab_src[i[0]]].append(num_words)
				else:
					self.word_idx[data.ivocab_src[i[0]]] = [num_words]
				self.word_s2t.append([data.ivocab_src[i[0]], data.ivocab_trg[i[1]]])
				num_words += 1
				print >> writefile, i[0] + ' ||| ' + i[1]
		print 'reserve', len(self.word_s2t), 'word features'
		self.feature_weight = numpy.ones((len(self.word_s2t),)) * config['feature_weight_word'] # ndarray [1,1,1....] the num of word_table's words   in ivocab

	def getFeatures(self, source, hypo, add_info = None):
		result = numpy.zeros((len(self.word_s2t),))
		for i in source:
			if self.word_idx.has_key(i):
				for j in self.word_idx[i]:
					if self.word_s2t[j][1] in hypo:
						result[j] = 1

		return result
	
class feature_phrase(featureList):

	def __init__(self, config, data):
		self.config = config
		self.data = data
		#load phrase table
		self.phrase_idx = {}
		self.phrase_s2t = []
		num_phrases = 0
		phrase_table = cPickle.load(open(config['phrase_table'], 'r'))
		writefile = open('phrase.txt', 'w')
		print 'total', len(phrase_table) ,'phrase entries'
		for i in phrase_table:
			source_words = i[0].split(' ')
			target_words = i[1].split(' ')
			if len(source_words) > config['max_phrase_length'] or len(target_words) > config['max_phrase_length']:
				continue
			nounk = True
			for j in range(len(source_words)):
				if data.ivocab_src.has_key(source_words[j]):
					source_words[j] = data.ivocab_src[source_words[j]]
				else:
					nounk = False
			for j in range(len(target_words)):
				if data.ivocab_trg.has_key(target_words[j]):
					target_words[j] = data.ivocab_trg[target_words[j]]
				else:
					nounk = False
			if not nounk:
				continue
			phrase_source = ' '.join([str(k) for k in source_words])
			if self.phrase_idx.has_key(phrase_source):
				self.phrase_idx[phrase_source].append(num_phrases)
			else:
				self.phrase_idx[phrase_source] = [num_phrases] 
			self.phrase_s2t.append([phrase_source, ' '.join([str(k) for k in target_words])])
			num_phrases += 1
			print >> writefile, i[0]+' ||| '+i[1]
		print 'reserve', len(self.phrase_s2t), 'phrase features'
		self.feature_weight = numpy.ones((len(self.phrase_s2t),)) * config['feature_weight_phrase']

	def getFeatures(self, source, hypo, add_info = None):
		result = numpy.zeros((len(self.phrase_s2t),))
		phrase_hypo = ' '.join([str(k) for k in hypo])
		for i in range(len(source)):
			for j in range(i + 1,min(len(source),i + self.config['max_phrase_length'])):
				phrase_source = ' '.join([str(k) for k in source[i:j]])
				if self.phrase_idx.has_key(phrase_source):
					for k in self.phrase_idx[phrase_source]:
						if self.phrase_s2t[k][1] in phrase_hypo:
							result[k] = 1

		return result
	 
class feature_length(featureList):

	def __init__(self, config, data):
		self.config = config
		self.data = data
		self.feature_weight = numpy.ones((1,)) * config['feature_weight_length_ratio']

	def getFeatures(self, x, y, add_info = None):
		#logging.info("sample: " +str(y))
		if len(x) * self.config['length_ratio'] > len(y):
			return numpy.asarray([1.0 * len(y) / (len(x) * self.config['length_ratio'])], dtype = 'float32')
		else:
			return numpy.asarray([1.0 * (len(x) * self.config['length_ratio']) / len(y)], dtype = 'float32')

class feature_treememory(featureList):

	def __init__(self, config, data):
		self.config = config
		self.data = data
		self.src_voc_dict = {v: k for k, v in data.ivocab_src.items()}
		self.trg_voc_dict = {v: k for k, v in data.ivocab_trg.items()}
		self.feature_weight = numpy.ones((2,)) * config['feature_weight_tree_memory']
		#self.parser = stanford.StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
		#chinese parser
		JAR_PATH = r'/home/tfzhang/berkeleyparser/BerkeleyParser-1.7.jar'
		ZH_GRM_PATH = r'/home/tfzhang/berkeleyparser/chn_sm5.gr'
		EN_GRM_PATH = r'/home/tfzhang/berkeleyparser/eng_sm6.gr'
		cp = os.environ.get("BERKELEY_PARSER_JAR", JAR_PATH)
		startup(cp)
		zhgr = os.environ.get("BERKELEY_PARSER_GRM", ZH_GRM_PATH)
		zhargs = {"gr": zhgr}
		self.zhopts = getOpts(dictToArgs(zhargs))
		self.parser = loadGrammar(self.zhopts)
		# english parser
		engr = os.environ.get("BERKELEY_PARSER_GRM", EN_GRM_PATH)
		enargs = {"gr": engr}
		#self.index=0
		self.enopts = getOpts(dictToArgs(enargs))
		self.enparser = loadGrammar(self.enopts)
		logging.info('feature treememory')
		self.zhstop_words = []
		self.enstop_words = []
		self.zhcorpus = open("/home/tfzhang/corpus/hansard/train.true.zh", 'r')
		self.zhfcorpus = self.zhcorpus.readlines()
		self.encorpus = open("/home/tfzhang/corpus/hansard/train.true.en", 'r')
		self.enfcorpus = self.encorpus.readlines()
		self.create_stop_words()
		self.zhrev_dict = self.build_index('zh')
		self.enrev_dict = self.build_index('en')
		logging.info("build revert index done!")


	def create_stop_words(self):
		with open("/home/tfzhang/corpus/hansard/vocab.zh", 'r') as fvocab:
			words = fvocab.readlines()
			for i in range(0, 158):  # freq mini 20000
				self.zhstop_words.append(words[i].split()[0])
		with open("/home/tfzhang/corpus/hansard/vocab.en", 'r') as fvocab:
			enwords = fvocab.readlines()
			for i in range(0, 274):  # freq mini 10000
				self.enstop_words.append(enwords[i].split()[0])
	def build_index(self,lang):
		mydict = {}
		if lang=='zh':
			for i, line in enumerate(self.zhfcorpus):
				for word in line.split():
					mydict.setdefault(word, []).append(i)
		if lang=='en':
			for i, line in enumerate(self.enfcorpus):
				for word in line.split():
					mydict.setdefault(word, []).append(i)
		return mydict
	def enpretrans(self,ntree):  # preprocess of tree
		posright = 0
		posspace = 0
		for i in range(len(ntree) - 1):
			if ntree[i] == "(":
				posleft = i
			elif ntree[i] == " ":
				posspace = i
			elif ntree[i] == ")":
				# print(posleft)
				if posright < posleft and 'DT' not in ntree[posleft:posspace] and 'IN' not in ntree[
																							  posleft:posspace] and 'CC' not in ntree[
																																posleft:posspace] and 'TO' not in ntree[
																																								  posleft:posspace] and 'MD' not in ntree[
																																																	posleft:posspace] and 'WRB' not in ntree[
																																																									   posleft:posspace] and 'WDT' not in ntree[
																																																																		  posleft:posspace] and 'WP' not in ntree[
																																																																											posleft:posspace] and 'WP\$' not in ntree[
																																																																																				posleft:posspace] and 'EX' not in ntree[
																																																																																												  posleft:posspace]:
					ntree = ntree[:posspace] + (i - posspace) * "#" + ntree[i:]
				# ntree = ntree[:posspace] +"#"+ ntree[posspace + 1:]  # remove the space
				posright = i

		ntree = re.sub('#', '', ntree)
		#logging.info(ntree)
		return ntree
	def zhpretrans(self,ntree):  # preprocess of tree
		posright = 0
		posspace = 0
		for i in range(len(ntree) - 1):
			if ntree[i] == "(":
				posleft = i
			elif ntree[i] == " ":
				posspace = i
			elif ntree[i] == ")":
				# print(posleft)
				if posright < posleft and '(LC ' not in ntree[posleft:posspace + 1] and 'CC' not in ntree[
																									posleft:posspace] and '(P ' not in ntree[
																																	   posleft:posspace + 1] and 'CS' not in ntree[
																																											 posleft:posspace]:
					ntree = ntree[:posspace] + (i - posspace) * "#" + ntree[i:]
				# ntree = ntree[:posspace] +"#"+ ntree[posspace + 1:]  # remove the space
				posright = i

		ntree = re.sub('#', '', ntree)
		#logging.info(ntree)
		return ntree
	def Deduplicate_NP(self,ntree):
		ntree = re.sub('\(NNP\)|\(NNS\)|\(NNPS\)', '(NN)', ntree)
		ntree = re.sub('\(IN ', '(IN_', ntree)
		ntree = re.sub('\(CC ', '(CC_', ntree)
		ntree = re.sub('\(TO ', '(TO_', ntree)
		ntree = re.sub('\(MD ', '(MD_', ntree)
		ntree = re.sub('\(WRB ', '(WRB_', ntree)
		ntree = re.sub('\(WDT ', '(WDT_', ntree)
		ntree = re.sub('\(WP ', '(WP_', ntree)
		ntree = re.sub('\(WP\$ ', '(WP$_', ntree)
		ntree = re.sub('\(EX ', '(EX_', ntree)
		ntree = re.sub('\(DT ', '(DT_', ntree)
		ntree = re.sub('\(LC ', '(LC_', ntree)
		ntree = re.sub('\(P ', '(P_', ntree)
		ntree = re.sub('\(CS ', '(CS_', ntree)
		node_list = ntree.split()
		node_list_len = len(node_list)
		for i in range(node_list_len):
			if '(NN)' in node_list[i] and '(NN)' in node_list[i + 1]:
				j = i
				while node_list[j + 1] == '(NN)':
					j += 1
				for m in range(i, j + 1):
					node_list[m] = '#jack#'
				i = j
		ntree = ''
		for node in node_list:
			if node != '#jack#':
				ntree += node + ' '
		ntree += '\n'
		return ntree
	def berkeleyparser(self,sen,lang):
		strIn = StringIO(sen.decode('utf-8'))
		if len(sen.split())>200:
			return "(())"
		strOut = StringIO()
		if lang=='zh':
			parseInput(self.parser, self.zhopts, inputFile=strIn, outputFile=strOut)
			result = self.enpretrans(strOut.getvalue())
		elif lang=='en':
			parseInput(self.enparser, self.enopts, inputFile=strIn, outputFile=strOut)
			result = self.zhpretrans(strOut.getvalue())
		#result = self.pretrans(strOut.getvalue())
		result = self.Deduplicate_NP(result)
		return result
	def stanfordparse(self,sen):

		result = list(self.parser.raw_parse(sen.decode("utf-8")))
		a=""
		for i in result:
			a += str(i)
		tree= self.pretrans(re.sub("\n  *", " ", a))
		return tree
	def transtree(self,oldtree):
		newt = oldtree[1:-1].decode("utf-8")
		nn = newt.count("(")  # calculate the num of nodes
		tlist = newt.split()
		C = Node(tlist[0][1:])
		tree = [0 for n in range(nn)]
		if nn == 0:
			C = Node("")
			return C, nn
		tree[0] = C
		j = 1
		for i in range(1, len(tlist)):
			if tlist[i][0] == "(":
				tree[j] = Node(re.sub('\)', '', tlist[i][1:]))
				tree[j - 1].addkid(tree[j])
			up = tlist[i].count(")")  # if the unit has ")",the node goes up
			if up > 0:
				j = j - up
			j += 1
		return C, nn

	def compare(self,curtree,lang):
		if not re.match("\(",curtree.decode('utf-8')):
			return 0
		trees = self.solrsearch(curtree, lang)
		fms =0
		compt1, len1 = self.transtree(curtree)
		if len(trees) == 0 or len1==0:
			return fms
		for tree in trees:
			#logging.info("tree:"+tree)
			compt2, len2 = self.transtree(tree)
			if len2==0:
				curfms=0
			else:
				dist = simple_distance(compt1, compt2)
				curfms = 1 - float(dist) / float(max(len1, len2))
			fms = max(fms, curfms)

		return fms

	def DiceSearch(self,cursen,lang):  # dice 100

		words_list = cursen.split()
		#logging.info(cursen)
		if lang=="en":


			words_list = [i for i in words_list if i not in self.enstop_words]

			len1 = len(words_list)
			a = set([])
			relist = []
			for word in words_list:
				if word in self.enrev_dict.keys():
					a = a.union(set(self.enrev_dict[word]))
		elif lang=="zh":
			words_list = [i for i in words_list if i not in self.zhstop_words]

			len1 = len(words_list)
			a = set([])
			relist = []
			for word in words_list:
				if word in self.zhrev_dict.keys():
					a = a.union(set(self.zhrev_dict[word]))
		if len(a) < 5:
			relist = list(a)
		# print("search less than 100")
		else:
			dice_dict = {}
			if lang=='zh':
				for i in a:
					sen_list = self.zhfcorpus[i].split()
					len2 = len(sen_list)
					cur_dice = round(2 * len(list(set(words_list).intersection(set(sen_list)))) / float(len1 + len2), 2)
					dice_dict[str(i)] = cur_dice
				dice_dict = sorted(dice_dict.items(), key=lambda d: d[1], reverse=True)
				i = 0
			elif lang=='en':
				for i in a:
					sen_list = self.enfcorpus[i].split()
					len2 = len(sen_list)
					cur_dice = round(2 * len(list(set(words_list).intersection(set(sen_list)))) / float(len1 + len2), 2)
					dice_dict[str(i)] = cur_dice
				dice_dict = sorted(dice_dict.items(), key=lambda d: d[1], reverse=True)
				i = 0
			for key in dice_dict:
				relist.append(key[0])
				i += 1
				if i > 5:
					break
		trees = []
		#logging.info("search sens num: "+str(len(relist)))
		# print(len(corpus))
		if lang=='zh':
			for i in relist:
				# print(i)
				text = re.sub('\(', '-LRB-', self.zhfcorpus[int(i)])
				text = re.sub('\)', '-RRB-', text)
				trees.append(self.berkeleyparser(text,lang))
		elif lang=='en':
			for i in relist:
				# print(i)
				text = re.sub('\(', '-LRB-', self.enfcorpus[int(i)])
				text = re.sub('\)', '-RRB-', text)
				trees.append(self.berkeleyparser(text,lang))

		return trees



	def solrsearch(self,curtree,lang):
		curtree = curtree.encode('utf-8')
		curtree = urllib2.quote(curtree)

		if lang=="zh":
			url = "http://localhost:8983/solr/ldcnews_xman_zh/select?q=target:\"" + curtree + "\""
		elif lang=="en":
			url = "http://localhost:8983/solr/ldcnews_xman_en/select?q=target:\"" + curtree + "\""
		result = urllib2.urlopen(url).read()
		response = etree.fromstring(result)
		result1 = response[1]
		trees = []
		for doc in result1:
			trees.append(doc[1][0].text)
		return trees


	def getFeatures(self, x, y, xfms,yfms,add_info = None):

		if add_info !=2:    # 0,1 represent training , 2 represent inference
			xsiml=float(xfms[0][:-1])
			if add_info== 1:
				seny = ""
				#logging.info(y)
				for index in y[:-1]:
					seny += self.trg_voc_dict[index] + " "
				tempy=seny
				seny = re.sub('\(', '-LRB-', seny)
				seny = re.sub('\)', '-RRB-', seny)
				ytree = self.berkeleyparser(seny,'en')
				ysiml = self.compare(ytree,'en')

			else:
				ysiml = float(yfms[0][:-1])
			return numpy.asarray([1.0 * xsiml, 1.0 * ysiml], dtype='float32')

		elif add_info==2:
			senx = ""
			for index in x[:-1]:
				senx+=self.src_voc_dict[index[0]]+" "
			seny = ""
			for index in y[:-1]:
				seny += self.trg_voc_dict[index] + " "
			senx = re.sub('\(', '-LRB-', senx)
			senx = re.sub('\)', '-RRB-', senx)
			xtree = self.berkeleyparser(senx,'zh')

			xsiml = self.compare(xtree,'zh')

			seny = re.sub('\(', '-LRB-', seny)
			seny = re.sub('\)', '-RRB-', seny)
			ytree = self.berkeleyparser(seny,'en')
			ysiml = self.compare(ytree, 'en')
			return numpy.asarray([1.0 * xsiml, 1.0 * ysiml], dtype='float32')


class feature_senmemory(featureList):

	def __init__(self, config, data):
		self.config = config
		self.data = data
		self.src_voc_dict = {v: k for k, v in data.ivocab_src.items()}
		self.trg_voc_dict = {v: k for k, v in data.ivocab_trg.items()}
		self.feature_weight = numpy.ones((2,)) * config['feature_weight_sen_memory']
		logging.info('Feature of Sentence Memory')
		self.zhstop_words = []
		self.enstop_words = []
		self.zhcorpus = open("/home/tfzhang/corpus/ldcnews/data/train.true.zh", 'r')
		self.zhfcorpus = self.zhcorpus.readlines()

		self.encorpus = open("/home/tfzhang/corpus/ldcnews/data/train.true.en", 'r')
		self.enfcorpus = self.encorpus.readlines()

		self.create_stop_words()
		self.zhfcorpus_fsw = []
		for line in self.zhfcorpus:
			words_list = line.split()
			words_list = [i for i in words_list if i not in self.zhstop_words]
			self.zhfcorpus_fsw.append(words_list)
		self.enfcorpus_fsw = []
		for line in self.enfcorpus:
			words_list = line.split()
			words_list = [i for i in words_list if i not in self.enstop_words]
			self.enfcorpus_fsw.append(words_list)
		self.zhrev_dict = self.build_index('zh')
		self.enrev_dict = self.build_index('en')
		logging.info("Build revert_index done!")


	def create_stop_words(self):
		with open("/home/tfzhang/corpus/hansard/vocab.zh", 'r') as fvocab:
			words = fvocab.readlines()
			for i in range(0, 333):  # freq mini 20000
				self.zhstop_words.append(words[i].split()[0])
		with open("/home/tfzhang/corpus/hansard/vocab.en", 'r') as fvocab:
			enwords = fvocab.readlines()
			for i in range(0, 274):  # freq mini 10000
				self.enstop_words.append(enwords[i].split()[0])
	def build_index(self,lang):
		mydict = {}
		if lang=='zh':
			for i, line in enumerate(self.zhfcorpus):
				for word in line.split():
					mydict.setdefault(word, []).append(i)
		if lang=='en':
			for i, line in enumerate(self.enfcorpus):
				for word in line.split():
					mydict.setdefault(word, []).append(i)
		return mydict

	def edit_distance(self,s, tmlist,lang):
		''' Levenshiten edit distance '''
		s = s.split()
		if lang=='zh':
			s = [m for m in s if m not in self.zhstop_words]
		elif lang=='en':
			s = [m for m in s if m not in self.enstop_words]
		lens = len(s)
		m = lens + 1
		compare = []
		'''add a "for" loop for 100 sentences '''
		for tmline in tmlist:  # tmlist:   [[index,source,target],[index,source,target].....]
			tms = tmline.split()
			if lang == 'zh':
				tms = [j for j in tms if j not in self.zhstop_words]
			elif lang == 'en':
				tms = [j for j in tms if j not in self.enstop_words]
			lentms = len(tms)
			n = lentms + 1
			edit = [[0] * n for i in range(m)]
			for i in range(0, m):
				edit[i][0] = i
			for i in range(0, n):
				edit[0][i] = i
			cost = 0
			for i in range(1, m):
				for j in range(1, n):
					if s[i - 1] == tms[j - 1]:
						cost = 0
					else:
						cost = 1
					edit[i][j] = min(edit[i - 1][j] + 1, edit[i][j - 1] + 1, edit[i - 1][j - 1] + cost)
			distance = edit[m - 1][n - 1]
			fms = 1 - float(distance) / float(max(lens, lentms))
			compare.append(fms)
		result = max(compare)

		return result  # result : [FMS,listoperate,index,tm_source,tm_target]

	def DiceSearch(self,cursen,lang):  # dice 100

		words_list = cursen.split()
		lensearch=len(words_list)
		a = []
		relist = []
		dice_dict = {}
		dice_dict1 = {}
		text = []
		#logging.info(cursen)
		if lang=="en":
			words_list = [i for i in words_list if i not in self.enstop_words]
			len1 = len(words_list)
			for word in words_list:
				if word in self.enrev_dict.keys():
					a += self.enrev_dict[word]
			key_list = Counter(a).most_common(700)
			i=0
			for key in key_list:
				sen_list = self.enfcorpus_fsw[key[0]]
				sen_list = [j for j in sen_list if j not in self.enstop_words]
				len2 = len(sen_list)
				if len2 > 50:
					continue
				cur_dice = round(2 * len(list(set(words_list).intersection(set(sen_list)))) / float(len1 + len2), 2)

				dice_dict[str(key[0])] = cur_dice
				i += 1
				if i > 50 and lensearch <= 10:
					break
			dice_dict = sorted(dice_dict.items(), key=lambda d: d[1], reverse=True)
			i = 0
			for key in dice_dict:
				relist.append(key[0])
				i += 1
				if i > 20:
					break
			for i in relist:
				text.append(self.enfcorpus[int(i)])
		elif lang=="zh":
			words_list = [i for i in words_list if i not in self.zhstop_words]
			len1 = len(words_list)
			for word in words_list:
				if word in self.zhrev_dict.keys():
					a += self.zhrev_dict[word]
			key_list = Counter(a).most_common(700)
			i = 0
			dice_dict = {}
			for key in key_list:
				sen_list = self.zhfcorpus_fsw[key[0]]
				sen_list = [j for j in sen_list if j not in self.zhstop_words]
				len2 = len(sen_list)
				if len2 > 50:
					continue
				cur_dice = round(2 * len(list(set(words_list).intersection(set(sen_list)))) / float(len1 + len2), 2)
				dice_dict[str(key[0])] = cur_dice
				i += 1
				if i > 50 and lensearch <= 10:
					break
			dice_dict = sorted(dice_dict.items(), key=lambda d: d[1], reverse=True)
			i = 0
			for key in dice_dict:
				relist.append(key[0])
				i += 1
				if i > 20:
					break
			for i in relist:
				text.append(self.zhfcorpus[int(i)])
		return text

	def getFeatures(self, x, y, xfms,yfms,add_info = None):

		if add_info !=2:    # 0,1 represent training , 2 represent inference

			xsiml=float(xfms[0][:-1])
			if add_info== 1:
				seny = ""
				for index in y[:-1]:
					seny += self.trg_voc_dict[index] + " "
				ytmsens=self.DiceSearch(seny,'en')
				if len(ytmsens)==0:
					ysiml=0
				else:
					ysiml = self.edit_distance(seny,ytmsens,'en')
			else:
				ysiml = float(yfms[0][:-1])
			#logging.info("sample_siml: "+str(ysiml))
			return numpy.asarray([1.0 * xsiml, 1.0 * ysiml], dtype='float32')

		elif add_info==2:
			senx = ""
			for index in x[:-1]:
				senx+=self.src_voc_dict[index[0]]+" "
			seny = ""
			for index in y[:-1]:
				seny += self.trg_voc_dict[index] + " "
			xtmsens = self.DiceSearch(senx, 'zh')
			if len(xtmsens) == 0:
				xsiml = 0
			else:
				xsiml = self.edit_distance(senx, xtmsens,'zh')
			ytmsens = self.DiceSearch(seny, 'en')
			if len(ytmsens) == 0:
				ysiml = 0
			else:
				ysiml = self.edit_distance(seny, ytmsens,'en')
			return numpy.asarray([1.0 * xsiml, 1.0 * ysiml], dtype='float32')



class feature_wordcount(featureList):

	def __init__(self, config, data):
		self.config = config
		self.data = data
		self.feature_weight = numpy.ones((1,)) * config['feature_weight_wordcount']

	def getFeatures(self, x, y, add_info = None):
		if add_info==1:
			return numpy.asarray([len(y)], dtype = 'float32')

def getMask(y,config):
	mask = numpy.ones((len(y),), dtype = 'float32')
	if config['index_eos_trg'] in y:
		mask[(y.index(config['index_eos_trg']) + 1):] = 0
	return mask

class feature_attention_coverage(featureListAttn):

	def __init__(self, config, data):
		self.config = config
		self.data = data
		self.feature_weight = numpy.ones((1,)) * config['feature_weight_attention_coverage']

	def getFeatures(self, x, y, add_info = None):
		#attention: len(y)*len(x)
		assert add_info
		print add_info[0].shape
		attention = add_info[0][:len(y)]
		#mask = getMask(y,self.config)
		#attn_sum = (attention*numpy.reshape(mask, (mask.shape[0],1))).sum(axis=0)
		attn_sum = attention.sum(axis = 0)
		attn_score = numpy.log(attn_sum.clip(0, 1)).sum()
		return numpy.asarray([attn_score], dtype = 'float32')


