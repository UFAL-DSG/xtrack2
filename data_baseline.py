from collections import Counter
from itertools import count
import collections
import json
import numpy as np

import data
import data_model


class BaselineSequence(dict):
    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        return self[key]

    def __init__(self, seq_id, source_dir):
        self.id = seq_id
        self.source_dir = source_dir
        self.data = []
        self.labels = []
        self.tags = collections.defaultdict(list)
        self.true_input = []

    def __repr__(self):
        return json.dumps(self.__dict__)


class DataBuilderBaseline(data.DataBuilder):
    seq_cls = BaselineSequence
    feature_cnts = Counter()
    feature_dict = {}

    def build(self, dialogs):
        res = super(DataBuilderBaseline, self).build(dialogs)
        # save only features that occur at least 5 times in the corpus

        res.vocab = {ftr: i for ftr, i in
                            zip(self.feature_cnts.keys(),count())}

        return res

    def _dump_seq_info(self, seq):
        pass

    def _process_dialog(self, dialog, seq):
        last_state = None

        features = None
        for msgs, state, actor in zip(dialog.messages,
                                      dialog.states,
                                      dialog.actors):
            actor_is_system = actor == data_model.Dialog.ACTOR_SYSTEM

            if actor_is_system:
                features = self._extract_features_system(msgs, seq)
            else:
                new_features = self._extract_features_user(msgs, seq)
                features.update(new_features)

            true_msg, _ = msgs[0]

            seq.data.append(features)
            self._append_label_to_seq(0.0, seq, state)

            for ftr in features:
                self.feature_cnts[ftr] += 1

            last_state = state

    def _extract_features_user(self, msgs, seq):
        features = {}
        for msg, msg_score in msgs[1:]:
            tokens = self._tokenize_msg(data_model.Dialog.ACTOR_USER, msg)
            tokens = [self._tag_token(token, seq) for token in tokens]

            tokens2 = zip(tokens, tokens[1:])
            tokens3 = [] #zip(tokens, tokens[1:], tokens[2:])

            features.update(self._tokens_to_features(tokens + tokens2 +
                                                     tokens3, msg_score))

        return features

    def _extract_features_system(self, msgs, seq):
        features = {}
        for msg, msg_score in msgs:
            tokens = self._tokenize_msg(data_model.Dialog.ACTOR_USER, msg)
            tokens = [self._tag_token(token, seq) for token in tokens]

            tokens2 = zip(tokens, tokens[1:])
            tokens3 = [] # zip(tokens, tokens[1:], tokens[2:])

            features.update(self._tokens_to_features(tokens + tokens2 +
                                                     tokens3, msg_score))


        return features

    def _tokens_to_features(self, tokens, score):
        features = {}
        for feature in tokens:
            if type(feature) is tuple:
                feature = '__'.join(feature)
            if not feature in features:
                features[feature] = np.exp(score)

        return features
