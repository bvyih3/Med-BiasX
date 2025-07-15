import os
import json
import torch
import random
import h5py
import numpy as np
import utils.utils as utils 
from torch.utils.data import Dataset
import utils.config as config
import copy
import _pickle as cPickle
import itertools
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

torch.utils.data.ConcatDataset.__getattr__ = lambda self, attr: getattr(self.datasets[0], attr)


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word, isQue):
        words = sentence
        if isQue:
            sentence = sentence.lower()
            sentence = sentence.replace(
                ',', '').replace('?', '').replace('\'s', ' \'s')
            words = sentence.split()
        tokens = []
        if add_word:
            if isQue:
                for w in words:
                    tokens.append(self.add_word(w))
            else:
                tokens.append(self.add_word(words))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        json.dump([self.word2idx, self.idx2word], open(path, 'w'))
        print('dictionary dumped to {}'.format(path))

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from {}'.format(path))
        word2idx, idx2word = json.load(open(config.dict_path, 'r'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'question': question['question'],
        'answer': answer,
        'answer_type': question['answer_type']}
    return entry


def _load_dataset(cache_path, name, img_id2val,ratio=1.0):
    """ Load entries. img_id2val: dict {img_id -> val} ,
        val can be used to retrieve image or features.
    """
    if 'vqace' in cache_path:
        prefix = name
        if name != 'train':
            if name == 'cou':
                prefix = 'counterexample'
        prefix += '_targets'
        question_path = os.path.join(config.main_path, prefix + '.json')
    else:
        question_path = os.path.join(config.main_path, name + '.json')
    print(question_path)
    questions = json.load(open(question_path, 'r'))
    questions = sorted(questions, key=lambda x: x['question_id'])
    answer_path = os.path.join(cache_path, '{}_target.json'.format(name))
    print(answer_path)
    answers = json.load(open(answer_path, 'r'))
    answers = sorted(answers, key=lambda x: x['question_id'])
    utils.assert_eq(len(questions), len(answers))
    if ratio < 1.0:
        print('--------ratio-----------',ratio)
        # sampling traing instance to construct smaller training set.
        index = random.sample(range(0, len(questions)), int(len(questions) * ratio))
        questions = [questions[i] for i in index]
        answers = [answers[i] for i in index]

    entries = []
    atype2idx = {}
    for question, answer in zip(questions, answers):
        if 'img_id' in question:
            question['image_id'] = question['img_id']
            question.pop('img_id')
        elif 'image id' in question:
            question['image_id'] = question['image id']
            question.pop('image id')
        if 'sent' in question:
            question['question'] = question['sent']
            question.pop('sent')
        if name == 'train':
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            if 'vqace' in question_path:
                question['image_id'] = int(question['image_id'].split('_')[2].lstrip('0'))
        if 'answer_type' not in question:
            question['answer_type'] = 'notype'
        if question['answer_type'] not in atype2idx:
            atype2idx[question['answer_type']] = len(atype2idx)
        img_id = str(question['image_id'])
        entries.append(_create_entry(img_id2val[img_id], question, answer))
    return entries, atype2idx


def _load_margin(cache_path, name):
    """ Load answer margin per question type.
    """

    mask_path = os.path.join(cache_path, '{}_margin.json'.format(name))
    print(mask_path)
    qt_dict = json.load(open(mask_path, 'r'))
    for qt in qt_dict:
        ans_num_dict = utils.json_keys2int(qt_dict[qt])
        ans = torch.tensor(list(ans_num_dict.keys()), dtype=torch.int64)
        portion = torch.tensor(list(ans_num_dict.values()), dtype=torch.float32)
        qt_dict[qt] = (ans, portion)

    mask_path = os.path.join(cache_path, '{}_freq.json'.format(name))
    qt_dict_freq = json.load(open(mask_path, 'r'))

    qt_cnt = {}
    qt_ans_cnt = {}
    ans_scale = {}

    for qt in qt_dict_freq:
        ans_num_dict = utils.json_keys2int(qt_dict_freq[qt])
        ans = torch.tensor(list(ans_num_dict.keys()), dtype=torch.int64)
        portion = torch.tensor(list(ans_num_dict.values()), dtype=torch.float32)
        qt_dict_freq[qt] = (ans, portion)
        qt_cnt[qt] = qt_cnt.get(qt, 0) + torch.sum(portion)
        qt_ans_cnt[qt] = len(ans)
        # if qt not in ans_scale:
            # ans_scale[qt] = {}
        # for ans, cnt in ans_num_dict.items():
            # ans_scale[qt][ans] = ans_scale[qt].get(ans, 0) + cnt
        for ans, cnt in ans_num_dict.items():
            ans_scale[ans] = ans_scale.get(ans, 0) + cnt
    total = sum(qt_cnt.values())
    # for qt in qt_cnt:
    #     for ans, cnt in ans_scale[qt].items():
    #         ans_scale[qt][ans] = cnt / qt_cnt[qt]
    for ans, cnt in ans_scale.items():
        ans_scale[ans] = cnt / total
    
    return qt_dict, qt_dict_freq, ans_scale


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, args):
        super(VQAFeatureDataset, self).__init__()
        self.dataset = args.dataset
        self.args = args
        if self.dataset == 'vqace':
            assert name in ['train', 'all', 'cou', 'easy', 'hard']
        else:
            print('name:', name)
            assert name in ['train', 'val', 'test']
        self.split = name
        config.dataset = self.dataset
        self.dictionary = dictionary

        # loading answer-label
        self.ans2label = json.load(open(os.path.join(config.cache_root,
            'traintest_ans2label.json'), 'r'))
        print(os.path.join(config.cache_root,
            'traintest_ans2label.json'))

        self.label2ans = json.load(open(os.path.join(config.cache_root,
            'traintest_label2ans.json'), 'r'))
        self.num_ans_candidates = len(self.ans2label)
        print('name:', name)
        print('self.num_ans_candidates:', self.num_ans_candidates)

        # object_hook=utils.json_keys2int if dataset == 'vqacp-v2' else None
        # loading image features
        if name== 'test':
            self.img_id2idx = json.load(open(os.path.join(config.ids_path,
                                                        'test36_imgid2idx.json'), 'r'))
            print(os.path.join(config.ids_path,
                               'test36_imgid2idx.json'))
        else:
            if self.dataset == 'vqace':
                self.img_id2idx = json.load(open(os.path.join(
                    config.ids_path, '{}36_imgid2idx.json'.format(
                        name)), 'r'))
                print(os.path.join(config.ids_path,
                                    '{}36_imgid2idx.json'.format(name)))
            else:
                self.img_id2idx = json.load(open(os.path.join(config.ids_path,
                                                        'train36_imgid2idx.json'), 'r'))
                print(os.path.join(config.ids_path,
                                    'train36_imgid2idx.json'))

        self.entries, self.atype2idx = _load_dataset(config.cache_root, name, self.img_id2idx,ratio=1.0)
        self.margins, self.freq, self.ans_scale = _load_margin(config.cache_root, name)
        for i in range(self.num_ans_candidates):
            if i not in self.ans_scale:
                self.ans_scale[i] = 0

        self.h5_path = os.path.join(config.rcnn_path, '{}_obj36.h5'.format(name))
        print(self.h5_path)

        # Convert list to dict (for evaluation)
        # self.id2datum = {
        #     datum['question_id']: {'labels': datum['answer']['labels'], 'scores': datum['answer']['scores']}
        #     for datum in self.entries
        # }
        self.transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5))]) 

        if 'MEVF' in args.base_model:
            # TODO: load images
            images_path = os.path.join(config.qa_path, 'images84x84.pkl')
            print('loading MAML image data from file: '+ images_path)
            self.maml_images_data = cPickle.load(open(images_path, 'rb'))
        # load image data for Auto-encoder module
            # TODO: load images
            images_path = os.path.join(config.qa_path, 'images128x128.pkl')
            print('loading DAE image data from file: '+ images_path)
            self.ae_images_data = cPickle.load(open(images_path, 'rb'))
        self.tokenize()
        self.tensorize()
        self.v_dim = config.output_features
        self.s_dim = config.num_fixed_boxes
        if 'MEVF' in args.base_model:
            self.v_dim = args.feat_dim * 2
        

    def tokenize(self, max_length=config.max_question_len):
        """ Tokenizes the questions.
            This will add q_token in each entry of the dataset.
            -1 represent nil, and should be treated as padding_idx in embedding.
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False, True)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        if 'MEVF' in self.args.base_model:
            self.maml_images_data = torch.from_numpy(self.maml_images_data)
            self.maml_images_data = self.maml_images_data.type('torch.FloatTensor')
            self.ae_images_data = torch.from_numpy(self.ae_images_data)
            self.ae_images_data = self.ae_images_data.type('torch.FloatTensor')
        if config.in_memory:
            self.features = torch.from_numpy(self.features)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def load_image(self, image_id):
        """ Load one image feature. """
        if not hasattr(self, 'image_feat'):
            self.image_feat = h5py.File(self.h5_path, 'r')
        features = self.image_feat['image_features'][image_id]
        boxes = self.image_feat['image_bb'][image_id]
        return features, boxes

    def __getitem__(self, index):
        entry = self.entries[index]
        
        question_id = entry['question_id']
        question = entry['q_token']
        answer = entry['answer']
        q_type = answer['question_type']
        ans_type = entry['answer_type']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)

        if self.args.base_model == 'SAN':
            if self.dataset in ['vqa-v2', 'vqacp-v1', 'vqacp-v2', 'vqace', 'gqaood']:
                name = (12 - len(str(entry['image_id']))) * '0' + str(entry['image_id'])
                image = config.resized_images_path.format('train', 'train', name)
                if not os.path.exists(image):
                    image = config.resized_images_path.format('val', 'val', name)
                image = Image.open(image).resize((224, 224)).convert('RGB')
            else:
                image = '{}/{}.jpg'.format(config.resized_images_path, entry['image_id'])
                image = Image.open(image).convert('RGB')
            image = self.transform(image)
            features = image
        elif 'MEVF' in self.args.base_model:
            image_data = [0, 0]
            maml_images_data = self.maml_images_data[entry['image']].reshape(84*84)
            image_data[0] = maml_images_data
            ae_images_data = self.ae_images_data[entry['image']].reshape(128*128)
            image_data[1] = ae_images_data
            features = image_data
        else:
            # Get image info
            features, boxes = self.load_image(entry['image'])

        margin_label, margin_score = self.margins[q_type]
        freq_label, freq_score = self.freq[q_type]

        betas = [0]
        torch.set_printoptions(profile="full")
        idx = 0
        eff = 1 - torch.float_power(betas[idx], freq_score)
        per0 = (1 - betas[idx]) / eff
        per0 = per0 / torch.sum(per0) * freq_score.shape[0]
        per0 = per0.float()

        target_margin = torch.zeros(self.num_ans_candidates)
        freq_margin0 = torch.zeros(self.num_ans_candidates)

        if labels is not None:
            target.scatter_(0, labels, scores)
            target_margin.scatter_(0, margin_label, margin_score)
            freq_margin0.scatter_(0, freq_label, per0)
        if q_type is None or q_type == 'None':
            q_type = 'notype'
        ans_type = self.atype2idx[ans_type]
        return features, question, target, target_margin, question_id, freq_margin0, q_type, ans_type

    def __len__(self):
        return len(self.entries)
    

    def tfidf_from_questions(names, args, dictionary, dataroot='data', target=['rad']):
        inds = [[], []] # rows, cols for uncoalesce sparse matrix
        df = dict()
        N = len(dictionary)
        def populate(inds, df, text):
            tokens = dictionary.tokenize(text, True, True)
            for t in tokens:
                df[t] = df.get(t, 0) + 1
            combin = list(itertools.combinations(tokens, 2))
            for c in combin:
                if c[0] < N:
                    inds[0].append(c[0]); inds[1].append(c[1])
                if c[1] < N:
                    inds[0].append(c[1]); inds[1].append(c[0])

        if 'rad' in target:
            for name in names:
                assert name in ['train', 'test']
                question_path = os.path.join(config.qa_path, name + '.json')
                # print(question_path)
                questions = json.load(open(question_path))
                for question in questions:
                    populate(inds, df, question['question'])

        # TF-IDF
        vals = [1] * len(inds[1])
        for idx, col in enumerate(inds[1]):
            assert df[col] >= 1, 'document frequency should be greater than zero!'
            vals[col] /= df[col]

        # Make stochastic matrix
        def normalize(inds, vals):
            z = dict()
            for row, val in zip(inds[0], vals):
                z[row] = z.get(row, 0) + val
            for idx, row in enumerate(inds[0]):
                vals[idx] /= z[row]
            return vals

        vals = normalize(inds, vals)

        tfidf = torch.sparse.FloatTensor(torch.LongTensor(inds), torch.FloatTensor(vals))
        # print('tfidf:', tfidf)
        tfidf = tfidf.coalesce()

        # Latent word embeddings
        glove_file = config.glove_path
        weights, word2emb = utils.create_glove_embedding_init(dictionary.idx2word[N:], glove_file)
        print('tf-idf stochastic matrix (%d x %d) is generated.' % (tfidf.size(0), tfidf.size(1)))

        return tfidf, weights
