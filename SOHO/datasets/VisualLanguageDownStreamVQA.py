import random
import json
import numpy as np
import pickle
import os

import torch
from torch.utils.data import Dataset
import commons
from commons.runner import get_dist_info

from commons import check_file_exist
from SOHO.utils import print_log
from .pipelines import Compose
from .registry import DATASETS
from .tokenization import BertTokenizer


@DATASETS.register_module
class VisualLanguageDownstreamVQA(Dataset):
    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix=None,
                 max_length=16,
                 sentence_group=3,
                 test_mode=False,
                 token_config="bert-base-uncased"
                 ):
        super(VisualLanguageDownstreamVQA, self).__init__()
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.sentence_group = sentence_group
        self.max_length=max_length
        self.test_mode=test_mode

        self.tokenizer = BertTokenizer.from_pretrained(token_config)

        source_data = self.read_anno()


        self.data_infors =source_data['data']
        self.ans2label = source_data['ans2label']
        self.label2ans = source_data['label2ans']
        self.num_ans_candidates = len(self.ans2label)

        self.debug_list=[]


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



    def pre_pipline(self,results):
        results['img_prefix'] = self.img_prefix
        results['language_tokens'] = results['img_info']['language_tokens']
        results['question_ids'] = results['img_info']['question_ids']
        results['language_attention'] = results['img_info']['language_attention']
        results['vqa_labels']=results['img_info']['vqa_labels']

    def prepare_train_img(self,index):
        data_info = self.create_item(index)
        results = dict(img_info=data_info)
        self.pre_pipline(results)
        return self.pipline(results)

    def prepare_test_img(self,index):
        data_info = self.create_item(index)
        results = dict(img_info=data_info)
        self.pre_pipline(results)
        return self.pipline(results)


    def _process_train_data(self):
        file_name = str(self.ann_file.split(".json")[0]) + "_vqa_g%d.pkl" % self.sentence_group
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

                pairs = []

                for index, pa in enumerate(item['vqa']):
                    if index % self.sentence_group == 0 and index != 0:
                        tmp = dict()
                        tmp['filename'] = filename
                        tmp['width'] = width
                        tmp['height'] = height
                        tmp['image_id'] = image_id
                        if len(pairs) > 0:
                            tmp['vqa'] = pairs
                            out_list.append(tmp)

                        pairs = []
                    pairs.append(pa)
                if len(pairs) > 0:
                    othe_num = self.sentence_group - len(pairs)
                    if othe_num <= len(item['vqa']):
                        random.shuffle(item['vqa'])
                        pairs.extend(item['vqa'][:othe_num])
                        tmp = dict()
                        tmp['filename'] = filename
                        tmp['width'] = width
                        tmp['height'] = height
                        tmp['image_id'] = image_id
                        tmp['vqa'] = pairs
                        out_list.append(tmp)
                    else:
                        for i in range(othe_num):
                            random.shuffle(item['vqa'])
                            pairs.append(item['vqa'][0])
                        tmp = dict()
                        tmp['filename'] = filename
                        tmp['width'] = width
                        tmp['height'] = height
                        tmp['image_id'] = image_id
                        tmp['vqa'] = pairs
                        out_list.append(tmp)

            print("all example", len(out_list) * self.sentence_group)
            self.data_infors = out_list
            pickle.dump(self.data_infors, open(os.path.join(self.data_root, file_name), 'wb'))

    def _process_test_data(self):
        out_list=[]
        self.q_id_list=[]
        for item in self.data_infors:
            tmp={
                "filename": item['file_name'],
                'image_id': item['image_id'],
                'height': item['height'],
                'width': item['width'],
                'question': item['question'],
                'question_id': item['question_id']
            }
            self.q_id_list.append(item['question_id'])
            out_list.append(tmp)
        self.data_infors = out_list


    def _langauge_process(self,cap):
        token_output = self.tokenizer.encode_plus(cap,add_special_tokens=True,padding='max_length',max_length=self.max_length,
                                               return_attention_mask=True,truncation=True)

        tokens = torch.tensor(token_output['input_ids'],dtype=torch.long).unsqueeze(dim=0)
        attention_mask = torch.tensor(token_output['attention_mask']).unsqueeze(dim=0)
        return tokens,attention_mask



    def create_item(self,index):
        if not self.test_mode:
            infors = self.data_infors[index].copy()
            language_list=[]
            vqa_labels = []
            token_attentions = []
            question_ids = []

            vqas = infors['vqa']
            for qa in vqas:
                language_token,language_attention=self._langauge_process(qa['question'])
                question_ids.append(qa['question_id'])

                language_list.append(language_token)
                token_attentions.append(language_attention)

                labels = torch.tensor(qa['labels'], dtype=torch.int64)
                scores = torch.tensor(qa['scores'])
                target = torch.zeros(self.num_ans_candidates)
                target.scatter_(0, labels, scores)
                vqa_labels.append(target)

            infors['language_tokens'] = torch.cat(language_list, dim=0)
            infors['language_attention'] = torch.cat(token_attentions, dim=0)
            infors['vqa_labels']=vqa_labels
            infors['question_ids']=question_ids

            return infors
        else:
            item = self.data_infors[index].copy()
            language_token, language_attention = self._langauge_process(item['question'])
            q_id = self.q_id_list.index(item['question_id'])
            item['question_ids'] = [torch.tensor(q_id),]
            item['language_tokens']=language_token
            item['language_attention']=language_attention
            item['vqa_labels']=[]
            return item

    def evaluate(self, results, logger=None,epoch=None, out_path=None,**kwargs):
        commons.mkdir_or_exist(out_path)
        tmp={"results":results,"index":self.q_id_list}
        pickle.dump(tmp, open(os.path.join(out_path, "outputs_{}.pkl".format(epoch)),'wb'),protocol=4)
        ids = results["ids"]
        preds = results["pred"]

        out_list=[]

        for id,pred in zip(ids,preds):
            q_id = self.q_id_list[int(id)]
            pred_index = np.argmax(pred, axis=0)
            answer = self.label2ans[pred_index]
            out_list.append({'question_id': q_id, 'answer': answer})

        commons.dump(out_list, os.path.join(out_path, "test_submit_{0}.json".format(str(epoch))))
        if logger is not None and logger !="silent":
            print_log("testing finished {} epoch".format(epoch),logger=logger)




