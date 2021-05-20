import platform
from functools import partial

from torch.utils.data import DataLoader

from commons.parallel import collate
from commons.runner import get_dist_info
from .sampler import DistributedGroupSampler, DistributedSampler, GroupSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(102400, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

# class DataLoaderX(DataLoader):
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())
#

def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     **kwargs):
    shuffle = kwargs.get('shuffle', True)
    if dist:
        rank, world_size = get_dist_info()
        if shuffle:
            sampler = DistributedGroupSampler(dataset, imgs_per_gpu,
                                              world_size, rank)
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=False)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = GroupSampler(dataset, imgs_per_gpu) if shuffle else None
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=True,
        **kwargs)
    # modify container
    # data_loader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     sampler=sampler,
    #     num_workers=num_workers,
    #     collate_fn=trim_collate,
    #     pin_memory=False,
    #     **kwargs)

    return data_loader
