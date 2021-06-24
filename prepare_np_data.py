import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from utils import add_special_tokens

tokenizer = add_special_tokens()

def make_npdata(file):
	""" Read a file, encode the text with GPT2 tokenizer, convert encoded text to numpy array and return it."""
	
	print('making_np_data_for_file:', file,end='\n\n')
	with open(file,'r') as f:
		data = json.load(f)
	text = tokenizer.encode(tokenizer.pad_token)*1024
	content = data['article'] + tokenizer.encode(tokenizer.sep_token) + data['abstract']
	text[:len(content)] = content
	print('completed_np_data_for_file:', file,end='\n\n')
	return np.array(text)


def call_back(results, file_name):
	""" Make a 2D numpy matrix and save it."""

	os.chdir('..')
	results = np.vstack(tuple(results))
	print('Shape of final array:', results.shape)
	print("Saving_np_files...")
	np.save(file_name,results)
	print("Successfully saved")
	print("\n")


if __name__ == '__main__':
	start = time.time()
	print("Process Started...")
	files = os.listdir(sys.argv[1])[:int(sys.argv[2])]
	dirname = os.path.split(sys.argv[1])[0]
	os.chdir(dirname)
	if 'CNN' in dirname:
		file_name = 'CNN_'+sys.argv[2]+'.npy'
	else:
		file_name = 'DM_'+sys.argv[2]+'.npy'
	pool = mp.Pool(processes=4)
	results = pool.map(make_npdata,files)
	call_back(results, file_name)
	pool.close()
	pool.join()
	print("total_time_taken: ", (time.time()-start)/60, " minutes")
