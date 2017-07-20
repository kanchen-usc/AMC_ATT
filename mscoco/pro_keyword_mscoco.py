import os, sys
import numpy as np
import json
import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer

min_count = 6
count_vect = CountVectorizer(min_df=min_count, stop_words='english')
keyword_dir = '../keyword/keywords_coco'

with open('%s/dataset.json'%keyword_dir) as fin:
	dataset_info = json.load(fin)
dataset_info = dataset_info['images']

def gen_count_vector(dataset_info, keyword_dir):
	keyword_all = []

	for img in dataset_info:
		if img['split'] == 'train' or img['split'] == 'test':
			keyword_file = '%s/keyword_%s/%s.json'%(keyword_dir, img['split'],img['filename'][:-4])
			with open(keyword_file) as keyword_fin:
				keyword_set = json.load(keyword_fin)
				keyword_set = keyword_set['tagging']['keywords']
				keyword_all.append(' '.join(keyword_set))

	X_train_counts = count_vect.fit_transform(keyword_all) 
	print('vocaulary size: %d'%(len(count_vect.vocabulary_)))
	pickle.dump(count_vect, open('count_vectorizer_coco_keyword_sk_%d.pkl'%min_count, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def gen_sparse_range_by_id(sparse_matrix, num_item, output_file):
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

if not os.path.exists('count_vectorizer_coco_keyword_sk_%d.pkl'%min_count):
	print 'Generating count vector'
	gen_count_vector(dataset_info, keyword_dir)

count_vect = np.load('count_vectorizer_coco_keyword_sk_%d.pkl'%min_count)
keyword_dict = count_vect.vocabulary_
analyzer = count_vect.build_analyzer()
img_keyword_rec = []

for img in dataset_info:
	if img['split'] == 'train' or img['split'] == 'test':
		keyword_file = '%s/keyword_%s/%s.json'%(keyword_dir, img['split'],img['filename'][:-4])
		with open(keyword_file) as keyword_fin:
			keyword_set = json.load(keyword_fin)
			keyword_set = keyword_set['tagging']['keywords']	
			keywords = ' '.join(keyword_set)
			keywords = analyzer(keywords)
			if len(keywords) > 0:
				keyword_ids = [keyword_dict.get(word) for word in keywords if word in keyword_dict]
				for word_id in keyword_ids:
					img_keyword_rec.append([img['imgid'], word_id])		

img_keyword_rec = np.array(img_keyword_rec).astype('int')
# np.save('img_keyword_rec_coco_sk_%d.npy'%min_count, img_keyword_rec)
gen_sparse_range_by_id(img_keyword_rec, np.max(img_keyword_rec[:, 0])+1, 'img_keyword_ind_coco_sk_%d.npy'%(min_count))

