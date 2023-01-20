import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import gol
import logging

from config import setup_config
from dataset.dataset import BaseDataset
from dataset.sampler import MetaTaskSampler
from torchvision import transforms
from model.registry import MODEL, FINETUNING
from model.finetune.dara_finetuning import DaraFinetuning
from utils import PerformanceMeter, TqdmHandler, AverageMeter, accuracy


class Tester(object):
    """Test a model from a config which could be a training config.
    """

    def __init__(self):
        self.config = setup_config()
        self.report_one_line = True
        self.logger = self.get_logger()
        self.set_gol()

        # set device. `config.experiment.cuda` should be a list of gpu device ids, None or [] for cpu only.
        self.device = self.config.experiment.cuda if isinstance(self.config.experiment.cuda, list) else []
        if len(self.device) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in self.device])
            self.logger.info(f'Using GPU: {self.device}')
        else:
            self.logger.info(f'Using CPU!')

        # build dataloader and model
        self.transformer = self.get_transformer(self.config.dataset.transformer)
        self.collate_fn = self.get_collate_fn()
        self.dataset = self.get_dataset(self.config.dataset)
        self.dataloader = self.get_dataloader(self.config.dataset)
        self.logger.info(f'Building model {self.config.model.name} ...')
        self.model = self.get_model(self.config.model)
        self.model = self.to_device(self.model, parallel=True)
        self.logger.info(f'Building model {self.config.model.name} OK!')
        self.finetuning = self.get_finetuning(self.config)

        # build meters
        self.performance_meters = self.get_performance_meters()
        self.average_meters = self.get_average_meters()

    def set_gol(self):
        gol._init()
        gol.set_value('is_ft', False)
        gol.set_value('use_transform', self.config.model.use_transform)

    def get_logger(self):
        logger = logging.getLogger()
        logger.handlers = []
        logger.setLevel(logging.INFO)

        screen_handler = TqdmHandler()
        screen_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
        logger.addHandler(screen_handler)
        return logger

    def get_performance_meters(self):
        return {
            metric: PerformanceMeter() for metric in ['acc']
        }

    def get_average_meters(self):
        return {
            meter: AverageMeter() for meter in ['acc']
        }

    def get_model(self, config):
        """Build model in config
        """
        name = config.name
        model = MODEL.get(name)(config)
        return model
    
    def load_model(self, config):
        """Load model in config
        """
        assert 'load' in config and config.load != '', 'There is no valid `load` in config[model.load]!'
        state_dict = torch.load(config.load, map_location='cpu')
        state_dict.pop('cat_mat')
        self.model.load_state_dict(state_dict, strict=False)
    
    def get_finetuning(self, config):
        """Get finetuning method
        """
        name = config.model.finetuning
        finetuning = FINETUNING.get(name)(config.dataset, self.model)
        return finetuning

    def get_transformer(self, config):
        return transforms.Compose([
            transforms.Resize(size=config.resize_size),
            transforms.CenterCrop(size=config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def get_collate_fn(self):
        return None

    def get_dataset(self, config):
        path = os.path.join(config.meta_dir, 'test.txt')
        return BaseDataset(config.root_dir, path, transform=self.transformer)

    def get_dataloader(self, config):
        way = config.way
        shot = config.shot
        trial = config.trail
        return DataLoader(self.dataset, batch_sampler=MetaTaskSampler(self.dataset, way=way, shot=shot, trial=trial), 
                          num_workers=config.num_workers, pin_memory=False)

    def to_device(self, m, parallel=False):
        if len(self.device) == 0:
            m = m.to('cpu')
        elif len(self.device) == 1 or not parallel:
            m = m.to(f'cuda:{self.device[0]}')
        else:
            m = m.cuda(self.device[0])
            m = torch.nn.DataParallel(m, device_ids=self.device)
        return m

    def get_model_module(self, model=None):
        """get `model` in single-gpu mode or `model.module` in multi-gpu mode.
        """
        if model is None:
            model = self.model
        if isinstance(model, torch.nn.DataParallel):
            return model.module
        else:
            return model

    def test(self):
        self.logger.info(f'Testing model from {self.config.model.load}')
        self.validate()
        self.performance_meters['acc'].update(self.average_meters['acc'].avg)
        self.report()

    def validate(self):
        val_bar = tqdm(self.dataloader, ncols=100, total=self.config.dataset.trail)
        for data in val_bar:
            self.task_validate(data)
            val_bar.set_description(f'Testing')
            val_bar.set_postfix(acc=self.average_meters['acc'].avg)
    

    def task_validate(self, data):
        self.load_model(self.config.model)
        images, labels = self.to_device(data['img']), self.to_device(data['label'])

        self.finetuning.meta_finetuning(images, self.config.finetuning.epoch)
        
        way = self.config.dataset.way
        shot = self.config.dataset.shot
        query_shot = self.config.dataset.query_shot

        query = images[way*shot:]
        target = torch.LongTensor([i // query_shot for i in range(query_shot * way)]).cuda()

        with torch.no_grad():
            logits = self.model(query)
            acc = accuracy(logits, target, 1)
            self.average_meters['acc'].update(acc, query.size(0))
        

    def report(self):
        metric_str = '  '.join([f'{metric}: {self.performance_meters[metric].current_value:.2f}'
                                for metric in self.performance_meters])
        self.logger.info(metric_str)


if __name__ == '__main__':
    tester = Tester()
    tester.test()
