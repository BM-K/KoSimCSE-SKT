import torch
from torch import nn


class BERT(nn.Module):
    def __init__(self, bert):
        super(BERT, self).__init__()
        self.bert = bert

    def forward(self, inputs, mode):

        if mode == 'train':
            anchor_attention_mask = self.gen_attention_mask(inputs['anchor']['source'],
                                                            inputs['anchor']['valid_length'])

            positive_attention_mask = self.gen_attention_mask(inputs['positive']['source'],
                                                              inputs['positive']['valid_length'])

            negative_attention_mask = self.gen_attention_mask(inputs['negative']['source'],
                                                              inputs['negative']['valid_length'])

            _, anchor_pooler = self.bert(input_ids=inputs['anchor']['source'],
                                         token_type_ids=inputs['anchor']['segment_ids'],
                                         attention_mask=anchor_attention_mask)

            _, positive_pooler = self.bert(input_ids=inputs['positive']['source'],
                                           token_type_ids=inputs['positive']['segment_ids'],
                                           attention_mask=positive_attention_mask)

            _, negative_pooler = self.bert(input_ids=inputs['negative']['source'],
                                           token_type_ids=inputs['negative']['segment_ids'],
                                           attention_mask=negative_attention_mask)

            return anchor_pooler, positive_pooler, negative_pooler

        else:
            sentence_1_attention_mask = self.gen_attention_mask(inputs['sentence_1']['source'],
                                                                inputs['sentence_1']['valid_length'])

            sentence_2_attention_mask = self.gen_attention_mask(inputs['sentence_2']['source'],
                                                                inputs['sentence_2']['valid_length'])

            _, sentence_1_pooler = self.bert(input_ids=inputs['sentence_1']['source'],
                                             token_type_ids=inputs['sentence_1']['segment_ids'],
                                             attention_mask=sentence_1_attention_mask)

            _, sentence_2_pooler = self.bert(input_ids=inputs['sentence_2']['source'],
                                             token_type_ids=inputs['sentence_2']['segment_ids'],
                                             attention_mask=sentence_2_attention_mask)

            return sentence_1_pooler, sentence_2_pooler

    def encode(self, inputs, device):

        attention_mask = self.gen_attention_mask(inputs['source'], inputs['valid_length'])

        _, embeddings = self.bert(input_ids=inputs['source'].to(device),
                                  token_type_ids=inputs['segment_ids'].to(device),
                                  attention_mask=attention_mask.to(device))

        return embeddings

    def gen_attention_mask(self, token_ids, valid_length):

        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1

        return attention_mask.float()
