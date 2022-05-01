import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import transforms
from tqdm import tqdm


class FlyData(Dataset):
    def __init__(self, phase, data_root):
        if phase not in ('train', 'test', 'valid'):
            raise (ValueError, 'phase must be one of train, valid or test')

        self.phase = phase
        self.data_root = data_root
        self.df = pd.DataFrame(self.index_df(self.phase, data_root))
        self.df = self.df.assign(id=self.df.index.values)

        self.unique_class = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {uc: idx for idx, uc in enumerate(self.unique_class)}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))


        self.dataid_to_filepath = self.df.to_dict()['filepath']
        self.dataid_to_class_id = self.df.to_dict()['class_id']

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        image = Image.open(self.dataid_to_filepath[item])
        image = self.transform(image)
        label = self.dataid_to_class_id[item]
        return image, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.unique_class)

    def index_df(self, phase, data_root):
        images = []
        data_folder = f'{data_root}/{phase}'
        set_len = 0
        print('Indexing {}...'.format(phase))
        for root, folders, files in os.walk(data_folder):
            set_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=set_len)
        for root, folders, files in os.walk(data_folder):
            if len(files) == 0:
                continue
            class_name = root.split('\\')[-1]
            for f in files:
                progress_bar.update(1)
                images.append({
                    'phase': phase,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })
        progress_bar.close()
        return images

class TaskSampler(Sampler):
    def __init__(self, dataset, episodes_per_epoch, n, k, q, num_tasks = 1):
        super(TaskSampler,self).__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset
        if num_tasks < 1:
            raise ValueError('num_tasks must be > 1')

        self.k = k
        self.n = n
        self.q = q
        self.num_tasks = num_tasks
    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []
            for task in range(self.num_tasks):
                episode_classes = np.random.choice(self.dataset.df['class_id'].unique(), size=self.k, replace=False)
                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]
                support_k = {k: None for k in episode_classes}
                for k in episode_classes:
                    support = df[df['class_id'] == k].sample(self.n)
                    support_k[k] = support

                    for i, s in support.iterrows():
                        batch.append(s['id'])

                for k in episode_classes:
                    query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].sample(self.q)
                    for i, q in query.iterrows():
                        batch.append(q['id'])
            yield np.stack(batch)

def prepare_nshot_task(batch, k, q):
    x, y = batch
    x = x.double().cuda()
    y = torch.arange(0, k, 1 / q).long().cuda()
    return x, y

# if __name__ == '__main__':
#     dataset = FlyData(phase='train', data_root='data/miniImageNet')
#     loader = DataLoader(
#         dataset,
#         batch_sampler=TaskSampler(dataset, episodes_per_epoch=100, n=1, k=10, q=5),
#         num_workers=0
#     )
#     for batch in loader:
#         x, y = prepare_nshot_task(batch, k=10, q=5)
#         print(x, y)