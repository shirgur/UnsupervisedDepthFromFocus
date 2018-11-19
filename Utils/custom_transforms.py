import torch
import random
import numpy as np
import cv2


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, depth):
        for t in self.transforms:
            image, depth = t(image, depth)
        return image, depth


class ConvertFromInts(object):
    def __call__(self, image, depth):
        return image.astype(np.float32), depth.astype(np.float32)


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, depth):
        if random.random() < 0.5:
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, depth


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, depth):
        if random.random() < 0.5:
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, depth


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, depth):
        if random.random() < 0.5:
            swap = self.perms[random.randint(0, len(self.perms)-1)]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, depth


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, depth):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, depth


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, depth):
        if random.random() < 0.5:
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, depth


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, depth):
        if random.random() < 0.5:
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, depth


class RandomMirror(object):
    def __call__(self, image, depth):
        if random.random() < 0.5:
            image = np.copy(np.fliplr(image))
            depth = np.copy(np.fliplr(depth))
        return image, depth


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, depth):
        im = image.copy()
        im, depth = self.rand_brightness(im, depth)
        if random.random() < 0.5:
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, depth = distort(im, depth)
        return self.rand_light_noise(im, depth)


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, image, depth):

        in_h, in_w, _ = image.shape
        x_scaling, y_scaling = np.random.uniform(1,1.15,2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        scaled_images = cv2.resize(image, (scaled_w, scaled_h))
        scaled_depth = cv2.resize(depth, (scaled_w, scaled_h))

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)

        cropped_images = scaled_images[offset_y:offset_y + in_h, offset_x:offset_x + in_w]
        cropped_depth = scaled_depth[offset_y:offset_y + in_h, offset_x:offset_x + in_w]

        return cropped_images, cropped_depth


class ClipRGB(object):
    def __call__(self, image, depth):
        image = image.astype(np.float32)
        return np.clip(image, 0, 255), depth


class ToTensor(object):

    def __call__(self, image, depth):
        # image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255
        # depth = torch.from_numpy(depth[:, :, 0]).unsqueeze(0).float() / 255
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        depth = torch.from_numpy(depth).float()

        return image, depth


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, depth):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), depth


class Augmentation(object):
    def __init__(self):

        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            RandomScaleCrop(),
            RandomMirror(),
            ClipRGB()
        ])

    def __call__(self, img, depth):
        return self.augment(img, depth)