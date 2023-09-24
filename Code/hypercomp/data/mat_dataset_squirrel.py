

from squirrel.iterstream.torch_composables import TorchIterable

from torch.utils.data import IterableDataset

from mat_to_squirrel.squirrel_ext import ConfigurableMessagePackDriver


class MatDatasetSquirrel(IterableDataset):
    """
    - root_dir/
        - test_2T3QHFKTCNUJFPWZ.gz
        - ...
        - train_2II06T9U9NMNSWIL.gz
        - ...
        - validation_0M7P8BYW485W6UYR.gz
        - ...
        """

    def __init__(self, root_dir, transform=None, split='train'):
        assert split == 'train' or split == 'val' or split == 'test', 'Invalid split'
        self.root_dir = root_dir
        self.split = split

        split = 'validation' if split == 'val' else split
        self.msg_pack_driver = ConfigurableMessagePackDriver(self.root_dir)

        self.transform = transform

        if transform:
            map_func = self.manipulate
        else:
            map_func = self.forward

        self.iter = self.msg_pack_driver.get_iter(split=split).tqdm().async_map(
            map_func, buffer=5).compose(TorchIterable)

    def __len__(self):
        return NotImplementedError()

    def __getitem__(self, index):
        return NotImplementedError()

    def __iter__(self):
        return self.iter.__iter__()

    @staticmethod
    def forward(arr):
        return arr.copy()

    def manipulate(self, arr):
        return self.transform(arr.copy())


# class MatDataset(Dataset):
#     """
#     - root_dir/
#         - 00000.mat
#         - 00001.mat
#         - 00002.mat
#         - ...
#     """
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.img_files = os.listdir(root_dir)
#         self.img_files.sort()

#         self.transform = transform

#     def __len__(self):
#         return len(self.img_files)

#     def __getitem__(self, index):
#         # get full image path
#         file_path = os.path.join(self.root_dir, self.img_files[index])
#         # read image
#         mat_dict = loadmat(file_path)
#         keys = list(mat_dict)
#         img_key = keys[-1]
#         img = mat_dict[img_key]
#         # convert dtype from float64 (double-precision) to float32 (float-precision)
#         img = img.astype(np.float32)
#         # apply transformations
#         if self.transform:
#             img = self.transform(img)
#         return img


# class MatDatasetFolder(MatDataset):
#     """
#     - root_dir/
#         - test/
#             - 00000.mat
#             - 00001.mat
#             - 00002.mat
#             - ...
#         - train/
#             - 00000.mat
#             - 00001.mat
#             - 00002.mat
#             - ...
#         - val/
#             - 00000.mat
#             - 00001.mat
#             - 00002.mat
#             - ...
#     """
#     def __init__(self, root_dir, transform=None, split='train'):
#         assert split == 'train' or split == 'val' or split == 'test', 'Invalid split'
#         self.split = split
#         super(MatDatasetFolder, self).__init__(root_dir + split, transform)


if __name__ == '__main__':
    import torch
    from tqdm import tqdm
    from math import ceil
    from torch.utils.data import DataLoader

    device = f'cuda:7' if torch.cuda.is_available() else 'cpu'

    squirrel_root = './fatih/squirrel/full/'

    dataset_split = 'train'
    img_transform = None
    batch_size = 16

    mat_dataset_squirrel = MatDatasetSquirrel(
        squirrel_root, transform=img_transform, split=dataset_split)

    dataloader_squirrel = DataLoader(
        mat_dataset_squirrel.iter,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )

    loop = tqdm(enumerate(dataloader_squirrel), total=ceil(
        2688 / dataloader_squirrel.batch_size), leave=True)

    for _, d in loop:
        d = d.to(device)
        print(d.shape)

        break
