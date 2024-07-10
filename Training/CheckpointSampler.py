from torch.utils.data import Sampler
from typing import Iterator
from torch.utils.data import DataLoader


class CheckpointSampler(Sampler[int]):
    def __init__(self, data_source: int, start_step: int, batch_size: int):
        super().__init__()
        self.data_source = data_source
        self.start_step = start_step
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        start_idx = self.start_step * self.batch_size
        return iter(range(start_idx, len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source) - (self.start_step * self.batch_size)


def dataloader_to_step(dataloader, step):
    # Create a new sampler starting from the checkpoint step
    new_sampler = CheckpointSampler(dataloader.dataset, step, dataloader.batch_size)

    # Create a new DataLoader with the updated sampler
    new_dataloader = DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        sampler=new_sampler,
        collate_fn=dataloader.collate_fn,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
    )

    return new_dataloader
