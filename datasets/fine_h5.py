import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os, sys
import logging
logging.basicConfig(level=logging.INFO)

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
logging.info(f"将{root_path}添加到了环境变量中")
sys.path.append(root_path)

# import augment.transforms_2d as transforms
import augment.transforms as transforms

class H5Dataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """
    def __init__(self, data, return_weight=False, transformer_config=None, phase='train'):
        assert data['mr'].shape[0] == data['weight'].shape[0] == data['unary'].shape[0] == data['mask'].shape[0]
        self.raw = data['mr']
        self.weight = data['weight']
        self.unary = data['unary']
        self.mask = data['mask']
        self.return_weight = return_weight
        self.transformer_config = transformer_config
        self.phase = phase

        if self.phase == 'train' and self.transformer_config is not None:
            self.transformer = transforms.get_transformer(transformer_config, mean=0, std=1, phase=phase)
            self.raw_transform = self.transformer.raw_transform()
            if self.return_weight:
                self.weight_transform = self.transformer.weight_transform()
            self.label_transform = self.transformer.label_transform()
    def __getitem__(self, index):
        if self.phase == 'train' and self.transformer_config is not None:
            raw = np.squeeze(self.raw[index])
            raw_transformed = self._transform_image(raw, self.raw_transform)
            raw_transformed = np.expand_dims(raw_transformed, axis=0)
            if self.return_weight:
                weight_transformed = self._transform_image(self.weight[index], self.weight_transform)
            label_transformed = self.mask[index]
            unary_transformed = self.unary[index]
            if self.return_weight:
                return raw_transformed.astype(np.float32), weight_transformed, unary_transformed, label_transformed.astype(np.long)
            else:
                return raw_transformed.astype(np.float32), unary_transformed, label_transformed.astype(np.long)
        else:
            if self.return_weight:
                return self.raw[index].astype(np.float32), self.weight[index], self.unary[index], self.mask[index].astype(np.long)
            else:
                return self.raw[index].astype(np.float32), self.unary[index], self.mask[index].astype(np.long)

    def __len__(self):
        return self.raw.shape[0]

    @staticmethod
    def _transform_image(dataset, transformer):
        return transformer(dataset)

def load_data(filePath, return_weight=True, batch_size=1, num_workers=0, shuffle=True, transformer_config=None):
    f = h5py.File(filePath, 'r')
    train_data = f['train']
    val_data = f['val']
    train_set = H5Dataset(data=train_data, transformer_config=transformer_config, return_weight=return_weight, phase='train')
    val_set = H5Dataset(data=val_data, transformer_config=transformer_config, return_weight=return_weight, phase='val')
    
    training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, pin_memory=False,
                                      shuffle=shuffle)

    val_data_loader = DataLoader(dataset=val_set, num_workers=num_workers, batch_size=batch_size, pin_memory=False,
                                      shuffle=False)

    return training_data_loader, val_data_loader, None, f

if __name__ == '__main__':
    inDir = '/mnt/ssd/dengyang/90_random_Verse/fine/in/h5py'
    filePath = os.path.join(inDir, 'fold1_data.h5')

    train_data_loader, val_data_loader, test_data_loader, f = load_data(filePath, batch_size=2)
    for i, t in enumerate(train_data_loader):
        print(i)
        mr, weight, unary, mask = t
        # import ipdb; ipdb.set_trace()
        print(mr.shape, weight.shape, unary.shape, mask.shape)
        

