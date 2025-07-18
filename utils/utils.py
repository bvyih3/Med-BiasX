import os
import json
import errno
import re
import collections
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import errno
import operator
import functools
from PIL import Image
import utils.config as config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import string_classes
from torch.utils.data.dataloader import default_collate
import math
import time
from utils.dataset import Dictionary, VQAFeatureDataset
from vqa_eval.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval

EPS = 1e-7
numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

def path_for(train=False, val=False, test=False, question=False, answer=False):
    assert train + val + test == 1
    assert question + answer == 1

    if train:
        split = 'train'
    elif val:
        split = 'val'
    else:
        split = 'test'

    fmt = split + '.json'

    return os.path.join(config.main_path, fmt)


def get_file(train=False, val=False, test=False, question=False, answer=False):
    """ Get the correct question or answer file."""
    _file = path_for(train=train, val=val, test=test,
                            question=question, answer=answer)
    with open(_file, 'r') as fd:
        _object = json.load(fd)
    return _object


def preprocess_answer(answer):
    """ Mimicing the answer pre-processing with evaluation server. """
    dummy_vqa = lambda: None
    dummy_vqa.getQuesIds = lambda: None
    vqa_eval = VQAEval(dummy_vqa, None)

    answer = vqa_eval.processDigitArticle(
            vqa_eval.processPunctuation(answer))
    answer = answer.replace(',', '')
    return answer


def assert_eq(real, expected):
    assert real == expected, '{} (true) vs {} (expected)'.format(real, expected)


def assert_array_eq(real, expected):
    assert (np.abs(real-expected) < EPS).all(), \
        '{} (true) vs {} (expected)'.format(real, expected)


def json_keys2int(x):
    return {int(k): v for k, v in x.items()}


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def load_imageid(folder):
    images = load_folder(folder, 'jpg')
    img_ids = set()
    for img in images:
        img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        img_ids.add(img_id)
    return img_ids


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def weights_init(m):
    """custom weights initialization."""
    cname = m.__class__
    if cname == nn.Linear or cname == nn.Conv2d or cname == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
    elif cname == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print('%s is not initialized.' % cname)

def init_net(net, net_file):
    if net_file:
        net.load_state_dict(torch.load(net_file))
    else:
        net.apply(weights_init)


def print_model(model, logger):
    print(model)
    nParams = 0
    for w in model.parameters():
        nParams += functools.reduce(operator.mul, w.size(), 1)
    if logger:
        logger.write('nParams=\t'+str(nParams))

def save_model(path, model, epoch, optimizer=None):
    model_dict = {
            'epoch': epoch,
            'model_state': model.state_dict()
        }
    if optimizer is not None:
        model_dict['optimizer_state'] = optimizer.state_dict()

    torch.save(model_dict, path)

# Select the indices given by `lengths` in the second dimension
# As a result, # of dimensions is shrinked by one
# @param pad(Tensor)
# @param len(list[int])
def rho_select(pad, lengths):
    # Index of the last output for each sequence.
    idx_ = (lengths-1).view(-1,1).expand(pad.size(0), pad.size(2)).unsqueeze(1)
    extracted = pad.gather(1, idx_).squeeze(1)
    return extracted

def trim_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    _use_shared_memory = True
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        out = None
        if 1 < batch[0].dim(): # image features
            max_num_boxes = max([x.size(0) for x in batch])
            if _use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = len(batch) * max_num_boxes * batch[0].size(-1)
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            # warning: F.pad returns Variable!
            return torch.stack([F.pad(x, (0,0,0,max_num_boxes-x.size(0))).data for x in batch], 0, out=out)
        else:
            if _use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [trim_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def append_bias(train_dset, eval_dset, answer_voc_size):
    """
        Compute the bias:
        The bias here is just the expected score for each answer/question type
    """
    # question_type -> answer -> total score
    question_type_to_probs = defaultdict(Counter)
    # question_type -> num_occurances
    question_type_to_count = Counter()
    for ex in train_dset.entries:
        ans = ex["answer"]
        q_type = ans["question_type"]
        question_type_to_count[q_type] += 1
        if ans["labels"] is not None:
            for label, score in zip(ans["labels"], ans["scores"]):
                question_type_to_probs[q_type][label] += score

    question_type_to_prob_array = {}
    for q_type, count in question_type_to_count.items():
        prob_array = np.zeros(answer_voc_size, np.float32)
        for label, total_score in question_type_to_probs[q_type].items():
            prob_array[label] += total_score
        prob_array /= count
        question_type_to_prob_array[q_type] = prob_array

    # Now add a `bias` field to each example
    for ds in [train_dset, eval_dset]:
        for ex in ds.entries:
            q_type = ex["answer"]["question_type"]
            ex["bias"] = question_type_to_prob_array[q_type]



class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)

def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    # glove_file = glove_file if args.use_TDIUC else os.path.join(args.TDIUC_dir, 'glove', glove_file.split('/')[-1])
    with open(glove_file, 'r', encoding='utf-8') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb

# --------------------FAIRSEQ functions---------------------------
def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)

def item(tensor):
    if hasattr(tensor, 'item'):
        return tensor.item()
    if hasattr(tensor, '__getitem__'):
        return tensor[0]
    return tensor

def clip_grad_norm_(tensor, max_norm):
    grad_norm = item(torch.norm(tensor))
    if grad_norm > max_norm > 0:
        clip_coef = max_norm / (grad_norm + 1e-6)
        tensor.mul_(clip_coef)
    return grad_norm

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

def get_size_of_largest_vqa_batch(dataloader):
    largest_v = None
    largest_b = None
    largest_q = None
    largest_a = None
    v, b, q, a = iter(dataloader).next()
    # ignore 1st dimension (batch size)
    largest_v = v.size()[1]
    largest_b = b.size()[1]
    largest_q = q.size()[1]
    largest_a = a.size()[1]
    for i, (v, b, q, a) in enumerate(dataloader):
        if largest_v > v.size()[1]:
            pass

def get_dummy_batch(args):
    pass

def as_minutes(seconds):
    minutes = math.floor(seconds / 60)
    seconds -= minutes * 60
    return '%dm %ds' % (minutes, seconds)

def time_since(since, percent):
    now = time.time()
    seconds = now - since
    elapsed_seconds = seconds / (percent)
    rest_seconds = elapsed_seconds - seconds
    return '%s (- %s)' % (as_minutes(seconds), as_minutes(rest_seconds))

def tfidf_loading(use_tfidf, w_emb, args):
    if use_tfidf:
        dict = Dictionary.load_from_file(os.path.join(config.qa_path, 'dictionary.json'))

        # load extracted tfidf and weights from file for saving loading time
        if os.path.isfile(os.path.join(config.qa_path, 'embed_tfidf_weights.pkl')) == True:
            print("Loading embedding tfidf and weights from file {}".format(os.path.join(config.qa_path, 'embed_tfidf_weights.pkl')))
            with open(os.path.join(config.qa_path ,'embed_tfidf_weights.pkl'), 'rb') as f:
                w_emb = torch.load(f)
            print("Load embedding tfidf and weights from file successfully")
        else:
            print("Embedding tfidf and weights haven't been saving before")
            tfidf, weights = VQAFeatureDataset.tfidf_from_questions(['train', 'test'], args, dict)
            # print(config.qa_path)
            w_emb.init_embedding(os.path.join(config.qa_path, 'glove6b_init.npy'), tfidf, weights)
            with open(os.path.join(config.qa_path ,'embed_tfidf_weights.pkl'), 'wb') as f:
                torch.save(w_emb, f)
            print("Saving embedding with tfidf and weights successfully")
    return w_emb

class Tracker:
    """ Keep track of results over time, while having access to
        monitors to display information about them.
    """
    def __init__(self):
        self.data = {}

    def track(self, name, *monitors):
        """ Track a set of results with given monitors under some name (e.g. 'val_acc').
            When appending to the returned list storage, use the monitors
            to retrieve useful information.
        """
        l = Tracker.ListStorage(monitors)
        self.data.setdefault(name, []).append(l)
        return l

    def to_dict(self):
        # turn list storages into regular lists
        return {k: list(map(list, v)) for k, v in self.data.items()}


    class ListStorage:
        """ Storage of data points that updates the given monitors """
        def __init__(self, monitors=[]):
            self.data = []
            self.monitors = monitors
            for monitor in self.monitors:
                setattr(self, monitor.name, monitor)

        def append(self, item):
            for monitor in self.monitors:
                monitor.update(item)
            self.data.append(item)

        def __iter__(self):
            return iter(self.data)

    class MeanMonitor:
        """ Take the mean over the given values """
        name = 'mean'

        def __init__(self):
            self.n = 0
            self.total = 0

        def update(self, value):
            self.total += value
            self.n += 1

        @property
        def value(self):
            return self.total / self.n

    class MovingMeanMonitor:
        """ Take an exponentially moving mean over the given values """
        name = 'mean'

        def __init__(self, momentum=0.9):
            self.momentum = momentum
            self.first = True
            self.value = None

        def update(self, value):
            if self.first:
                self.value = value
                self.first = False
            else:
                m = self.momentum
                self.value = m * self.value + (1 - m) * value
