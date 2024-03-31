from torch.utils.data import Dataset
from PIL import Image


class SingleImageDataset(Dataset):
    def __init__(
        self,
        image: Image.Image,
    ) -> None:
        super().__init__()
        self.image = image

    def __getitem__(self, index):
        return self.image
