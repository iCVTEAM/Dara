import numpy as np
from copy import deepcopy

from torch.utils.data.sampler import BatchSampler


class BalancedBatchSampler(BatchSampler):

    def __init__(self, dataset, n_classes, n_samples):
        self.labels = np.array(dataset.images['label'])
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size


# sampler used for meta-testing
class MetaTaskSampler(BatchSampler):
    def __init__(self, dataset, way, shot, query_shot=16, trial=1000):

        class2id = {}

        for index in range(len(dataset)):
            item = dataset.images.iloc[index]
            class_id = item['label']
            if class_id not in class2id:
                class2id[class_id] = []
            class2id[class_id].append(index)

        self.class2id = class2id
        self.way = way
        self.shot = shot
        self.trial = trial
        self.query_shot = query_shot

    def __iter__(self):

        way = self.way
        shot = self.shot
        trial = self.trial
        query_shot = self.query_shot

        class2id = deepcopy(self.class2id)
        list_class_id = list(class2id.keys())

        for i in range(trial):

            id_list = []

            np.random.shuffle(list_class_id)
            picked_class = list_class_id[:way]

            for cat in picked_class:
                np.random.shuffle(class2id[cat])

            for cat in picked_class:
                id_list.extend(class2id[cat][:shot])
            for cat in picked_class:
                id_list.extend(class2id[cat][shot:(shot + query_shot)])

            yield id_list
