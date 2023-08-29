import os
from functools import partial

import torch
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, \
    DistilBertForSequenceClassification


def get_bert(model_name_or_path, num_classes, max_seq_length):
    model = BertForSequenceClassification.from_pretrained(model_name_or_path,
                                                          num_labels=num_classes,
                                                          cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])

    def forward_generator(self, x):
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        token_type_ids = x['token_type_ids']

        # BEGIN ENCODER
        # equivalent to: x = self.encoder(x)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask,
                                                                                 input_shape=input_ids.size())
        x = self.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        # go through encoder blocks
        for block in self.bert.encoder.layer:
            x = block(
                x,
                extended_attention_mask,
            )[0]

            x = yield x, None

        # END OF ENCODER
        # classifier token
        pooled_output = (
            self.bert.pooler(x) if self.bert.pooler is not None else x
        )
        pooled_output = self.dropout(pooled_output)
        x = self.classifier(pooled_output)
        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)

    org_forward = model.forward

    def forward_wrapper(self, x):
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        token_type_ids = x['token_type_ids']
        return org_forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits

    model.forward = partial(forward_wrapper, model)
    model.input_sequences = {
        'input_ids': max_seq_length,
        'token_type_ids': max_seq_length,
        'attention_mask': max_seq_length,
    }
    model.number_of_classes = num_classes

    return model


def get_bert_base(num_classes, max_seq_length):
    return get_bert('bert-base-uncased', num_classes, max_seq_length)


def get_bert_large(num_classes, max_seq_length):
    return get_bert('bert-large-uncased', num_classes, max_seq_length)


def get_distilbert_base(num_classes, max_seq_length):
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                num_labels=num_classes,
                                                                cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])

    def forward_generator(self, x):
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']

        # BEGIN ENCODER
        # equivalent to: x = self.encoder(x)
        x = self.distilbert.embeddings(
            input_ids=input_ids,
        )
        # go through encoder blocks
        for block in self.distilbert.transformer.layer:
            x = block(
                x,
                attention_mask,
            )[-1]

            x = yield x, None

        # END OF ENCODER
        # classifier token
        x = x[:, 0]  # (bs, dim)
        x = self.pre_classifier(x)  # (bs, dim)
        x = torch.nn.ReLU()(x)  # (bs, dim)
        x = self.dropout(x)  # (bs, dim)
        x = self.classifier(x)  # (bs, num_labels)

        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)

    org_forward = model.forward

    def forward_wrapper(self, x):
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        return org_forward(input_ids=input_ids, attention_mask=attention_mask).logits

    model.forward = partial(forward_wrapper, model)
    model.input_sequences = {
        'input_ids': max_seq_length,
        'attention_mask': max_seq_length,
    }
    model.number_of_classes = num_classes

    return model


def get_roberta_base(num_classes, max_seq_length):
    model = RobertaForSequenceClassification.from_pretrained('robert-base',
                                                             num_labels=num_classes,
                                                             cache_dir=os.environ['TRANSFORMERS_CACHE_DIR'])

    def forward_generator(self, x):
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']

        # BEGIN ENCODER
        # equivalent to: x = self.encoder(x)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask,
                                                                                 input_shape=input_ids.size())
        x = self.roberta.embeddings(
            input_ids=input_ids,
        )
        # go through encoder blocks
        for block in self.roberta.encoder.layer:
            x = block(
                x,
                attention_mask=extended_attention_mask,
            )[0]

            x = yield x, None

        # END OF ENCODER
        # classifier token
        pooled_output = (
            self.roberta.pooler(x) if self.roberta.pooler is not None else x
        )
        x = self.classifier(pooled_output)
        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)

    org_forward = model.forward

    def forward_wrapper(self, x):
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        return org_forward(input_ids=input_ids, attention_mask=attention_mask).logits

    model.forward = partial(forward_wrapper, model)
    model.input_sequences = {
        'input_ids': max_seq_length,
        'attention_mask': max_seq_length,
    }
    model.number_of_classes = num_classes

    return model
