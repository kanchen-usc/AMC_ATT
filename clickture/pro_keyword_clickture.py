import os, sys
import numpy as np
import time
import json
import cPickle as pickle

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(min_df=6, stop_words='english')

file_list_lite = 'clickture-lite.lst'
data_dir_lite = '../keyword/keywords_lite'
file_list_dev = 'clickture-dev.lst'
data_dir_dev = '../keyword/keywords_dev'

def gen_count_vector(file_list, data_dir):
	keywords_all = []
	with open(file_list) as fin:
		for fid, file_name in enumerate(fin.readlines()):
			json_file = '%s/%s.json'%(data_dir, file_name.strip())
			with open(json_file) as fin2:
				data = json.load(fin2)
				keywords = data['tagging']['keywords']
				keywords = ' '.join(keywords)
				keywords_all.append(keywords)

	X_train_counts = count_vect.fit_transform(keywords_all) 
	print('vocaulary size: %d'%(len(count_vect.vocabulary_)))
	pickle.dump(count_vect, open('count_vectorizer_clickture_dev_sk_6.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def gen_compact_range_by_id(sparse_matrix, num_item, output_file):
    cur_id = 0
    cur_start = 0
    sparse_range = (-1) * np.ones((num_item, 2))
    for line_id, item_id in enumerate(sparse_matrix[:, 0]):
        if not item_id == cur_id:
            sparse_range[cur_id, :] = [cur_start, line_id-1]
            cur_start = line_id
            cur_id = item_id
    sparse_range[cur_id, :] = [cur_start, len(sparse_matrix[:, 0])-1]
    np.save(output_file, sparse_range.astype(int))

if not os.path.exists('count_vectorizer_clickture_dev_sk_6.pkl'):
	print 'Generating count vector'
	# We focus on evaluating test set keywords, discard those only appearing in training (lite) set
	gen_count_vector(file_list_dev, data_dir_dev)

count_vect = np.load('count_vectorizer_clickture_dev_sk_6.pkl')
analyzer = count_vect.build_analyzer()
word_dict = count_vect.vocabulary_

print 'Encoding Lite Set'
img_keyword_rec = []
with open(file_list_lite) as fin:
	for fid, file_name in enumerate(fin.readlines()):
		img_id = int(file_name.strip().split('/')[-1])
		json_file = '%s/%s.json'%(data_dir_lite, file_name.strip())
		with open(json_file) as fin2:
			data = json.load(fin2)
			keywords = data['tagging']['keywords']
			keywords = ' '.join(keywords)
			keywords = analyzer(keywords)
			if len(keywords) > 0:
				keyword_ids = [word_dict.get(word) for word in keywords if word in word_dict]
				for word_id in keyword_ids:
					img_keyword_rec.append([img_id, word_id])

img_keyword_rec = np.array(img_keyword_rec).astype('int')
# np.save('img_keyword_rec_clickture_lite_sk_6.npy', img_keyword_rec)
gen_compact_range_by_id(img_keyword_rec, np.max(img_keyword_rec[:, 0])+1, 'img_keyword_ind_clickture_lite_sk_6_sparse.npy')

print 'Encoding Dev Set'
img_keyword_rec = []
with open(file_list_dev) as fin:
	for fid, file_name in enumerate(fin.readlines()):
		img_id = int(file_name.strip().split('/')[-1])
		json_file = '%s/%s.json'%(data_dir_dev, file_name.strip())
		with open(json_file) as fin2:
			data = json.load(fin2)
			keywords = data['tagging']['keywords']
			keywords = ' '.join(keywords)
			keywords = analyzer(keywords)
			if len(keywords) > 0:
				keyword_ids = [word_dict.get(word) for word in keywords if word in word_dict]
				for word_id in keyword_ids:
					img_keyword_rec.append([img_id, word_id])

img_keyword_rec = np.array(img_keyword_rec).astype('int')
# np.save('img_keyword_rec_clickture_dev_sk_6.npy', img_keyword_rec)
gen_compact_range_by_id(img_keyword_rec, np.max(img_keyword_rec[:, 0])+1, 'img_keyword_ind_clickture_dev_sk_6_sparse.npy')
