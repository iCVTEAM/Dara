import torch
import torch.nn as nn

from model.registry import FINETUNING

@FINETUNING.register
class DaraFinetuning():
    
    def __init__(self, config, model):
        self.model = model

        self.way = config.way
        self.shot = config.shot
        self.query_shot = config.query_shot

        self.criterion = nn.CrossEntropyLoss()

    def meta_finetuning(self, data, epoch):
        support = data[:self.way*self.shot]

        self.prototypical_feature_reprojection_stage1(support, epoch)
        self.Instance_level_feature_recalibration(support, data)
        self.prototypical_feature_reprojection_stage2(support, epoch)
        
    def prototypical_feature_reprojection_stage1(self, support, epoch):
        optimizer_1st = torch.optim.SGD(self.model.feature_extractor.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)

        support_feat = self.model.get_feature_map(support).view(self.way, self.shot, self.model.resolution, self.model.d)
        self.model.cat_mat = nn.Parameter(support_feat.mean(1))

        support_view = support.view(self.way, self.shot, *support.shape[-3:])
        for t in range(epoch):
            indexes = torch.randperm(self.shot)
            s_idxes = indexes[:1]
            if self.shot > 1:
                q_idxes = indexes[1:]
            else:
                q_idxes = indexes[:1]
            
            # dividing pseudo support and pseudo query
            part_quft = support_view[:, q_idxes].view(self.way * (self.shot - 1), *support.shape[-3:])
            part_spft = support_feat[:, s_idxes].squeeze(1)
            self.model.cat_mat = nn.Parameter(part_spft)
            if self.shot > 1:
                self.step(self.shot-1, part_quft, optimizer_1st)
            else:
                self.step(self.shot, part_quft, support, optimizer_1st)
    
    def prototypical_feature_reprojection_stage2(self, support, epoch):
        optimizer_2nd = torch.optim.SGD([self.model.cat_mat], lr=0.01, momentum=0.9, weight_decay=0.001, nesterov=True)
        for t in range(epoch):
            self.step(self.shot, support, optimizer_2nd)
    
    def Instance_level_feature_recalibration(self, support, data):
        weight = torch.zeros(self.way, self.shot).cuda()
        with torch.no_grad():
            support_feat = self.model.get_feature_map(support).view(self.way, self.shot, self.model.resolution, self.model.d)
            data_feat = self.model.get_feature_map(data).view(self.way, self.shot + self.query_shot, self.model.resolution, self.model.d)
            for cls in range(self.way):
                dis = self.compute_d(support_feat[cls], data_feat[cls]).mean(1)
                dis = 1 / dis
                weight[cls] += (dis / dis.sum())
        weighted_support_feat = support_feat * weight[:, :, None, None]
        self.model.cat_mat = nn.Parameter(weighted_support_feat.mean(1))
    
    def compute_d(self, f1, f2):
        t1 = f1.unsqueeze(1).expand(len(f1), len(f2), f1.shape[1], f1.shape[2])
        t2 = f2.unsqueeze(0).expand(len(f1), len(f2), f2.shape[1], f2.shape[2])
        d = (t1 - t2).pow(2).sum((2, 3))
        return d

    def step(self, q_shot, query, optimizer):
        target = torch.LongTensor([i // q_shot for i in range(q_shot * self.way)]).cuda()
        
        # forward
        outputs = self.model(query)
        loss = self.criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()