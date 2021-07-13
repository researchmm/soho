import json
import os
import pickle
import random

import torch
import numpy as np
from torch.utils.data import Dataset

from commons import check_file_exist
from .pipelines import Compose
from .registry import DATASETS
from .tokenization import BertTokenizer

@DATASETS.register_module
class VisualLanguagePretrainDataset(Dataset):
    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix=None,
                 max_length=16,
                 sentence_group=3,
                 max_predictions_per_seq=3,
                 masked_lm_prob=0.15,
                 index_file=None,
                 use_qa=False,
                 use_answer=False,
                 test_mode=False,
                 token_config="bert-base-uncased"
                 ):
        super(VisualLanguagePretrainDataset, self).__init__()
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.test_mode = test_mode
        self.max_length = max_length
        self.sentence_group = sentence_group
        self.mlm_probability = masked_lm_prob
        self.use_answer = use_answer
        self.use_qa = use_qa
        self.tokenizer = BertTokenizer.from_pretrained(token_config)
        self.index_file = index_file
        if self.index_file is not None:
            self.index_data = json.load(open(self.index_file))
        else:
            self.index_data = None

        self.data_infors = self.read_anno()
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infors = [self.data_infors[i] for i in valid_inds]
            self._process_train_data()
        else:
            self._process_test_data()

        self._set_group_flag()
        self.pipline = Compose(pipeline)


    def read_anno(self):
        return json.load(open(os.path.join(self.data_root, self.ann_file)))

    def _filter_imgs(self, min_size=32):
        """
        Filter images too small
        :param min_size:
        :return:
        """
        valid_inds = []
        for i, img_info in enumerate(self.data_infors):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """
        Set flag according to iamge sapect ration
        Image with aspect ration greater than 1 will be set as group 1. otherwise group 0
        :return:
        """
        self.flag = np.zeros(len(self.data_infors), dtype=np.int)
        for i in range(len(self.data_infors)):
            img_info = self.data_infors[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_annoter(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        while True:
            if self.test_mode:
                data = self.prepare_test_img(idx)
            else:
                data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_annoter(idx)
                continue
            return data
    def __len__(self):
        return len(self.data_infors)

    def indextofolder(self, img_prefix, file_name):
        if self.index_data:
            folder = self.index_data[file_name]
            return os.path.join(self.data_root, folder)
        else:
            return img_prefix

    def alignName(self, file_name):
        if self.index_data:
            folder = self.index_data[file_name]
            return os.path.join(folder, file_name)
        else:
            return file_name

    def pre_pipline(self,results):
        results['img_prefix'] = self.indextofolder(self.img_prefix, results['img_info']['filename'])
        results['language_tokens'] = results['img_info']['language_tokens']
        results['input_image_id'] = results['img_info']['input_image_id']
        results['mask_labels'] = results['img_info']['mask_labels']
        results['next_label'] = results['img_info']['next_label']
        results['language_attention'] = results['img_info']['language_attention']
        results['img_target']=results['img_info']['img_target']

    def prepare_train_img(self,index):
        data_info = self.create_item(index)
        results = dict(img_info=data_info)
        self.pre_pipline(results)
        return self.pipline(results)

    def prepare_test_img(self,index):
        pass

    def _process_train_data(self):
        file_name = str(self.ann_file.split(".json")[0]) + "_vl_g%d.pkl" % self.sentence_group
        if check_file_exist(os.path.join(self.data_root, file_name)) and os.path.getsize(
                os.path.join(self.data_root, file_name)) > 0:
            self.data_infors = pickle.load(open(os.path.join(self.data_root, file_name), 'rb'))
        else:
            out_list = []
            for item in self.data_infors:
                filename = item['file_name']
                image_id = item['image_id']
                height = item['height']
                width = item['width']
                labels = item['labels']

                pairs = []
                if item.get('questionAns', None) is None or self.use_qa is False:
                    captions = item['captions']
                else:
                    if self.use_answer:
                        captions = item['captions'] + item['questionAns']
                    else:
                        captions = item['captions'] + item['question']
                for index, pa in enumerate(captions):
                    if index % self.sentence_group == 0 and index != 0:
                        tmp = dict()
                        tmp['filename'] = filename
                        tmp['width'] = width
                        tmp['height'] = height
                        tmp['image_id'] = image_id
                        tmp['labels']=labels
                        if len(pairs) > 0:
                            tmp['caption'] = pairs
                            out_list.append(tmp)

                        pairs = []
                    pairs.append(pa)
                if len(pairs) > 0:
                    othe_num = self.sentence_group - len(pairs)
                    if othe_num <= len(item['captions']):
                        random.shuffle(item['captions'])
                        pairs.extend(item['captions'][:othe_num])
                        tmp = dict()
                        tmp['filename'] = filename
                        tmp['width'] = width
                        tmp['height'] = height
                        tmp['image_id'] = image_id
                        tmp['caption'] = pairs
                        tmp['labels'] = labels
                        out_list.append(tmp)
                    else:
                        for i in range(othe_num):
                            random.shuffle(item['captions'])
                            pairs.append(item['captions'][0])
                        tmp = dict()
                        tmp['filename'] = filename
                        tmp['width'] = width
                        tmp['height'] = height
                        tmp['image_id'] = image_id
                        tmp['caption'] = pairs
                        tmp['labels'] = labels
                        out_list.append(tmp)

            print("all example", len(out_list) * self.sentence_group)
            self.data_infors = out_list
            pickle.dump(self.data_infors, open(os.path.join(self.data_root, file_name), 'wb'))

    def _process_test_data(self):
        pass

    def mask_tokens(self,inputs):
        """
        Prepare masked tokens inputs/labels for masked langauge modeling:80% MASK, 10% random, 10% original
        :param inputs:
        :return:
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling."
                " Remove the --mlm flag if you want to use this tokenizer."
            )
        inputs = inputs.unsqueeze(dim=0)
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training
        # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels



    def _langauge_process(self,cap):
        token_output = self.tokenizer.encode_plus(cap,add_special_tokens=True,padding='max_length',max_length=self.max_length,
                                               return_attention_mask=True,truncation=True)

        tokens = torch.tensor(token_output['input_ids'],dtype=torch.long)
        attention_mask = torch.tensor(token_output['attention_mask']).unsqueeze(dim=0)
        mask_tokens,mask_labels = self.mask_tokens(tokens)
        return mask_tokens,mask_labels,attention_mask



    def create_item(self,index):
        if not self.test_mode:
            infors = self.data_infors[index].copy()
            img_id = infors['image_id']
            # if len(infors['labels'])==0:
            #     infors['labels'].append(0)
            img_cls_label = torch.tensor(infors['labels'], dtype=torch.int64)
            img_target = torch.zeros(1601)
            # label_num = len(infors['labels'])
            # cls_score = 1.0/label_num if label_num >0 else 1.0

            img_target.scatter_(0, img_cls_label, 1.0)

            language_list=[]
            mask_labels = []
            token_attentions = []
            img_ids = []
            next_labels=[]
            name = infors['filename'].split(".jpg")[0]
            captions = infors['caption']
            for cap in captions:
                next_prob = random.random()
                next_flag = False
                if next_prob<0.5:
                    while True:
                        tmp_index = random.randint(0, len(self.data_infors) - 1)
                        tmp_id = self.data_infors[tmp_index]['image_id']
                        if tmp_id !=img_id:
                            cap = self.data_infors[tmp_index]['caption'][0]
                            next_flag = True
                            break

                language_token,mask_label,language_attention = self._langauge_process(cap)
                if not next_flag:
                    next_labels.append(1)
                else:
                    next_labels.append(0)
                language_list.append(language_token)
                mask_labels.append(mask_label)
                token_attentions.append(language_attention)
                img_ids.append(img_id)

            infors['language_tokens']= torch.cat(language_list,dim=0)
            infors['mask_labels']=torch.cat(mask_labels,dim=0)
            infors['language_attention']=torch.cat(token_attentions,dim=0)
            infors['next_label']=np.array(next_labels,dtype=np.int)
            infors['input_image_id']=img_ids
            infors['img_target']=img_target
            return infors
        else:
            pass



