import os
import sys
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt


def make_dir(path):
	if not os.path.exists(path):
		os.mkdir(path)
		os.chdir(path)

#calculates no of words in cnn/dm articles
def calc_article_sizes(file_name, name):
	
	max_len = 0
	article_sizes = {}
	print("Calculating",name, "Article Sizes......")
	for i,file in enumerate(os.listdir(file_name)):
		file = os.path.join(os.getcwd(),file_name,file)
		with open(file,'r',encoding='utf-8') as f:
			txt = f.read().split()
		txt_len = len(txt)
		article_sizes[os.path.basename(file)] = txt_len
		if max_len<txt_len:
			max_len = txt_len
			max_len_filename = os.path.basename(file)
		if i%100==0:
			print(i+1, " files read")
	return max_len, max_len_filename, article_sizes


if __name__ == '__main__':

	start = time.time()
	if sys.argv[1].startswith("cnn"):
		name = "CNN"
	else:
		name = "DM"

	make_dir("./"+name)
	max_len, max_len_filename, article_sizes = calc_article_sizes(sys.argv[1], name)
	sorted_article_values = np.array(sorted(article_sizes.values()))
	article_sizes = dict(sorted(article_sizes.items(), key=lambda item:item[1]))
	print("saving_article_files_sizes_info...")
	os.chdir(name)
	with open(name+"_file_size.pickle", 'wb') as f:
		pickle.dump(article_sizes, f)

	#plot the distribution of articles sizes
	plt.hist(sorted_article_values,color='blue',bins=6, edgecolor = 'black')
	plt.title(name+"_Files_Distribution_By_Size(no. of words)")
	plt.xlabel('No Of Words')
	plt.ylabel('Files')
	plt.show()
	plt.savefig(name+" files distribution by length")

	print('max_length_of_article_in_article: ',max_len, " and file_name is ", max_len_filename)
	print("total_time_taken",(time.time()-start)/60, " minutes")
	print("mean_length_of_article_articles: ", sum(article_sizes.values())/len(article_sizes))
	print("max_10_lengths_of_article_articles:", sorted_article_values[-50:])
	print("number_of_articles_greater_than_1000_words: ", sum(sorted_article_values>1000))
	print("number_of_articles_greater_than_1500_words: ", sum(sorted_article_values>1500))
	print("number_of_articles_greater_than_2000_words: ", sum(sorted_article_values>2000))