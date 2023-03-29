import torch
import torchvision.transforms.functional as F


class CoyoteCrop(torch.nn.Module):
    """Crops the given image at the coyote. If the image is torch Tensor, it is
    expected to have [..., H, W] shape, where ... means an arbitrary number of
    leading dimensions.
    """

    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.
        Returns:
            PIL Image or Tensor: Cropped image.
        """
        
        return F.crop(img, 0, 0, 100, 100)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"
