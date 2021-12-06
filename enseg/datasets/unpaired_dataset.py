from enseg.datasets.builder import DATASETS, build_dataset
from enseg.datasets.cityscapes import CityscapesDataset
from enseg.datasets.nightcity import NightCityDataset
from random import randint
from .custom import CustomDataset
from enseg.datasets.builder import build_dataset


@DATASETS.register_module()
class UnpairedDataset(NightCityDataset):
    def __init__(self, aux_dataset=None, **kwargs):
        super().__init__(**kwargs)
        self.aux_dataset = build_dataset(aux_dataset, dict(test_mode=kwargs.get("test_mode", False)))

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            aux_idx = randint(0, len(self.aux_dataset) - 1)
            return (
                self.prepare_train_img(idx),
                self.aux_dataset.prepare_train_img(aux_idx),
            )
