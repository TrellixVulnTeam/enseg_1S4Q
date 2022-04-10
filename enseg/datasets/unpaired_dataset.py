from enseg.datasets.builder import DATASETS, build_dataset
from enseg.datasets.nightcity import NightCityDataset
from random import randint
from enseg.datasets.builder import build_dataset


@DATASETS.register_module()
class UnpairedDataset(NightCityDataset):
    def __init__(self, aux_dataset=None, **kwargs):
        super().__init__(**kwargs)
        if aux_dataset != None:
            self.aux_dataset = build_dataset(
                aux_dataset, dict(test_mode=kwargs.get("test_mode", False))
            )
        else:
            self.aux_dataset = None

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            A = self.prepare_train_img(idx)
            if self.aux_dataset is not None:
                aux_idx = randint(0, len(self.aux_dataset) - 1)
                B = self.aux_dataset.prepare_train_img(aux_idx)
            else:
                B = A
            return (A, B)
