
import numpy
import torch
import logging
import gluonnlp as nlp
from torch.utils.data import DataLoader, Dataset
from KoBERT.kobert.utils import get_tokenizer
from KoBERT.kobert.pytorch_kobert import get_pytorch_kobert_model

logger = logging.getLogger(__name__)


class ModelDataLoader(Dataset):
    def __init__(self, file_path, args, metric, tokenizer, vocab, type):
        self.type = type
        self.args = args
        self.vocab = vocab
        self.metric = metric

        """NLI"""
        self.anchor = []
        self.positive = []
        self.negative = []

        """STS"""
        self.label = []
        self.sentence_1 = []
        self.sentence_2 = []

        #  -------------------------------------
        self.bert_tokenizer = tokenizer

        self.transform = nlp.data.BERTSentenceTransform(
            self.bert_tokenizer, max_seq_length=self.args.max_len, pad=True, pair=False)

        self.file_path = file_path

        """
        [CLS]: 2
        [PAD]: 1
        [UNK]: 0
        """
        self.init_token = self.vocab.cls_token
        self.pad_token = self.vocab.padding_token
        self.unk_token = self.vocab.unknown_token

        self.init_token_idx = self.vocab.token_to_idx[self.init_token]
        self.pad_token_idx = self.vocab.token_to_idx[self.pad_token]
        self.unk_token_idx = self.vocab.token_to_idx[self.unk_token]

    def load_data(self, type):

        with open(self.file_path) as file:
            lines = file.readlines()

            for line in lines:
                self.data2tensor(line, type)

        if type == 'train':
            assert len(self.anchor) == len(self.positive) == len(self.negative)
        else:
            assert len(self.sentence_1) == len(self.sentence_2) == len(self.label)

    def data2tensor(self, line, type):
        split_data = line.split('\t')

        if type == 'train':
            anchor, positive, negative = split_data
            anchor = self.transform([anchor])
            positive = self.transform([positive])
            negative = self.transform([negative])

            self.anchor.append(anchor)
            self.positive.append(positive)
            self.negative.append(negative)

        else:
            sentence_1, sentence_2, label = split_data
            sentence_1 = self.transform([sentence_1])
            sentence_2 = self.transform([sentence_2])

            self.sentence_1.append(sentence_1)
            self.sentence_2.append(sentence_2)
            self.label.append(float(label.strip())/5.0)

    def __getitem__(self, index):

        if self.type == 'train':
            inputs = {'anchor': {
                'source': torch.LongTensor(self.anchor[index][0]),
                'valid_length': torch.tensor(self.anchor[index][1]),
                'segment_ids': torch.LongTensor(self.anchor[index][2])
                },
                      'positive': {
                'source': torch.LongTensor(self.positive[index][0]),
                'valid_length': torch.tensor(self.positive[index][1]),
                'segment_ids': torch.LongTensor(self.positive[index][2])
                },
                      'negative': {
                'source': torch.LongTensor(self.negative[index][0]),
                'valid_length': torch.tensor(self.negative[index][1]),
                'segment_ids': torch.LongTensor(self.negative[index][2])
                }}
        else:

            inputs = {'sentence_1': {
                'source': torch.LongTensor(self.sentence_1[index][0]),
                'valid_length': torch.tensor(self.sentence_1[index][1]),
                'segment_ids': torch.LongTensor(self.sentence_1[index][2])
                },
                      'sentence_2': {
                'source': torch.LongTensor(self.sentence_2[index][0]),
                'valid_length': torch.tensor(self.sentence_2[index][1]),
                'segment_ids': torch.LongTensor(self.sentence_2[index][2])
                },
                      'label': torch.FloatTensor([self.label[index]])}

        inputs = self.metric.move2device(inputs, self.args.device)

        return inputs

    def __len__(self):
        if self.type == 'train':
            return len(self.anchor)
        else:
            return len(self.label)


# Get train, valid, test data loader and BERT tokenizer
def get_loader(args, metric):
    bert_model, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    tokenizer = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    path_to_train_data = args.path_to_data + '/' + args.task + '/' + args.train_data
    path_to_valid_data = args.path_to_data + '/' + args.task + '/' + args.valid_data
    path_to_test_data = args.path_to_data + '/' + args.task + '/' + args.test_data

    if args.train == 'True' and args.test == 'False':
        train_iter = ModelDataLoader(path_to_train_data, args, metric, tokenizer, vocab, type='train')
        valid_iter = ModelDataLoader(path_to_valid_data, args, metric, tokenizer, vocab, type='valid')

        train_iter.load_data('train')
        valid_iter.load_data('valid')

        loader = {'train': DataLoader(dataset=train_iter,
                                      batch_size=args.batch_size,
                                      shuffle=True),
                  'valid': DataLoader(dataset=valid_iter,
                                      batch_size=args.batch_size,
                                      shuffle=True)}

    elif args.train == 'False' and args.test == 'True':
        test_iter = ModelDataLoader(path_to_test_data, args, metric, tokenizer, vocab, type='test')
        test_iter.load_data('test')

        loader = {'test': DataLoader(dataset=test_iter,
                                     batch_size=args.batch_size,
                                     shuffle=True)}

    else:
        loader = None

    return bert_model, loader, tokenizer


def convert_to_tensor(corpus, transform):
    tensor_corpus = []
    tensor_valid_length = []
    tensor_segment_ids = []
    for step, sentence in enumerate(corpus):
        cur_sentence, valid_length, segment_ids = transform([sentence])

        tensor_corpus.append(cur_sentence)
        tensor_valid_length.append(numpy.array([valid_length]))
        tensor_segment_ids.append(segment_ids)

    inputs = {'source': torch.LongTensor(tensor_corpus),
              'segment_ids': torch.LongTensor(tensor_segment_ids),
              'valid_length': torch.tensor(tensor_valid_length)}

    return inputs


def example_model_setting(model_ckpt):

    from model.simcse.bert import BERT

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    bert_model, vocab = get_pytorch_kobert_model()
    tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)
    transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=50, pad=True, pair=False)

    model = BERT(bert_model)

    model.load_state_dict(torch.load(model_ckpt)['model'])
    model.to(device)
    model.eval()

    return model, transform, device


if __name__ == '__main__':
    get_loader('test')
