import random
from collections import OrderedDict

def split_train_test_fa(fasta_file):
	
	#count the number of samples
	count = 0
	for line in open(fasta_file):
		if line[0] == '>':
			count += 1
			
	print count
	
	trainSplit = int(count*0.7)
	validSplit = int(trainSplit + count*0.1)
	testSplit = int(validSplit + count*0.1)

	# load and code sequences
	seq_vecs = OrderedDict()
	seq = ''
	for line in open(fasta_file):
		if line[0] == '>':
			if seq:
				seq_vecs[header] = seq
			#
			header = line.rstrip()
			seq = ''
		else:
			seq += line.rstrip()
	#
	if seq:
		seq_vecs[header] = seq
	
	
	items = seq_vecs.items()
	random.shuffle(items)
	dict = OrderedDict(items)
	
	i = 0
	for header in dict.keys():
		i += 1
		if i <= trainSplit:
			with open("neg-train.fa", "a") as ftrain:				
				ftrain.write(header) 
				ftrain.write('\n')
				ftrain.write(dict[header])
				ftrain.write('\n')
		elif i <= validSplit:
                        with open("neg-valid.fa", "a") as ft:
                                ft.write(header)
                                ft.write('\n')
                                ft.write(dict[header])
                                ft.write('\n')
		elif i <= testSplit:
			with open("neg-test1.fa", "a") as ftest:				
				ftest.write(header) 
				ftest.write('\n')
				ftest.write(dict[header])
				ftest.write('\n')
		else:
			with open("neg-test2.fa", "a") as fvalid:				
				fvalid.write(header) 
				fvalid.write('\n')
				fvalid.write(dict[header])
				fvalid.write('\n')
		
		
split_train_test_fa("negative-cdhit.fa")
