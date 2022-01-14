import os
import os.path
import random
from copy import deepcopy
from typing import Callable, cast, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None):
    class_to_instances = {}
    instances_and_class = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        instances = []
        fnames = sorted(os.listdir(target_dir))
        for fname in fnames:
            path = os.path.join(target_dir, fname)
            if is_valid_file(path):
                instances.append(path)
        class_to_instances[target_class] = instances
        instances_and_class.extend(zip(instances, [class_to_idx[target_class]] * len(instances)))
    num_instances = len(instances_and_class)
    return class_to_instances, instances_and_class, num_instances


class DatasetFolder(Dataset):
    """
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.extensions = extensions
        classes, class_to_idx = self._find_classes(self.root)
        self.classes = classes  # idx -> class
        self.class_to_idx = class_to_idx  # class -> idx

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

            root/dog/xxx.png
            root/dog/xxy.png
            root/dog/xxz.png

            root/cat/123.png
            root/cat/nsdf3.png
            root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        images (list): List of (images path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader=pil_loader, extensions=IMG_EXTENSIONS, transform=None, target_transform=None,
                 is_valid_file=None):
        super().__init__(root, loader, extensions, transform, target_transform)
        _, images, num_images = make_dataset(self.root, self.class_to_idx, extensions, is_valid_file)
        if len(images) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)
        self.images = images
        self.num_images = num_images
        self.targets = [s[1] for s in images]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.images[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return self.num_images


class ImageFolder_(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

            root/dog/xxx.png
            root/dog/xxy.png
            root/dog/xxz.png

            root/cat/123.png
            root/cat/nsdf3.png
            root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

    Attributes:
       classes (list): List of the class names sorted alphabetically.
       class_to_idx (dict): Dict with items (class_name, class_index).
       class_to_images (dict): Dict with items (class_name, listdir(class))
       num_images (list): The number of images in the dataset
    """

    def __init__(self, root, loader=pil_loader, extensions=IMG_EXTENSIONS, transform=None, target_transform=None,
                 is_valid_file=None):
        super().__init__(root, loader, extensions, transform, target_transform)
        class_to_images, _, num_images = make_dataset(self.root, self.class_to_idx, extensions, is_valid_file)
        if num_images == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)
        self.class_to_images = class_to_images  # class -> list(image_paths)
        self.num_images = num_images

        assert len(self.classes) == len(self.class_to_idx) == len(self.class_to_images)

    def __len__(self):
        return self.num_images


class ImageFolderG(ImageFolder_):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: ((image1, image2), similarity of 0. or 1.)
        """
        cls1 = cls2 = random.choice(self.classes)

        get_same_class = random.randint(0, 1)
        if not get_same_class:
            while cls1 == cls2:
                cls2 = random.choice(self.classes)

        image1_path = random.choice(self.class_to_images[cls1])
        image2_path = random.choice(self.class_to_images[cls2])
        image1 = self.loader(image1_path)
        image2 = self.loader(image2_path)

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return (image1, image2), get_same_class.__float__()

    @staticmethod
    def collate_fn(batch):
        batch_image_pairs = []
        batch_similarity = []

        for image_pair, similarity in batch:
            batch_image_pairs.append(torch.stack(image_pair, dim=0))
            batch_similarity.append(similarity)

        return torch.stack(batch_image_pairs, dim=0), torch.as_tensor(batch_similarity).unsqueeze(dim=1)


class ImageFolderUnsupG(ImageFolder):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: ((image1, image2), similarity of 0. or 1.)
        """
        path, target = random.choice(self.images)
        image1 = self.loader(path)

        get_same_class = random.randint(0, 1)
        if get_same_class:
            image2 = deepcopy(image1)
        else:
            path, target = random.choice(self.images)
            image2 = self.loader(path)

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return (image1, image2), get_same_class.__float__()

    @staticmethod
    def collate_fn(batch):
        batch_image_pairs = []
        batch_similarity = []

        for image_pair, similarity in batch:
            batch_image_pairs.append(torch.stack(image_pair, dim=0))
            batch_similarity.append(similarity)

        return torch.stack(batch_image_pairs, dim=0), torch.as_tensor(batch_similarity).unsqueeze(dim=1)


class ImageFolderD(ImageFolder_):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: ((image1, image2, similarity of 0. or 1.), label of 0. or 1.)
        """
        cls1 = cls2 = random.choice(self.classes)

        get_same_class = random.randint(0, 1)
        if not get_same_class:
            while cls1 == cls2:
                cls2 = random.choice(self.classes)

        set_right_similarity = random.randint(0, 1)

        similarity = get_same_class if set_right_similarity else 1 - get_same_class

        image1_path = random.choice(self.class_to_images[cls1])
        image2_path = random.choice(self.class_to_images[cls2])
        image1 = self.loader(image1_path)
        image2 = self.loader(image2_path)

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return ((image1, image2), similarity.__float__()), set_right_similarity.__float__()

    @staticmethod
    def collate_fn(batch):
        batch_image_pairs = []
        batch_similarity = []
        batch_labels = []

        for (image_pair, similarity), label in batch:
            batch_image_pairs.append(torch.stack(image_pair, dim=0))
            batch_similarity.append(similarity)
            batch_labels.append(label)

        return (torch.stack(batch_image_pairs, dim=0), torch.as_tensor(batch_similarity).unsqueeze(dim=1)), \
               torch.as_tensor(batch_labels).unsqueeze(dim=1)


class ImageFolderRepr(ImageFolder_):
    def __init__(self, root, num_shot, batch_size, loader=pil_loader, extensions=IMG_EXTENSIONS, transform=None,
                 target_transform=None, is_valid_file=None):
        super().__init__(root, loader, extensions, transform, target_transform, is_valid_file)
        self.num_shot = num_shot
        self.batch_size = batch_size

    def __getitem__(self, index):
        class_to_repr = {}
        for cls in self.class_to_images.keys():
            repr = random.sample(self.class_to_images[cls], self.num_shot)
            for img in repr:
                repr_images = self.transform(pil_loader(img)).unsqueeze(0).expand(self.batch_size, -1, -1,
                                                                                  -1)  # [B, 3, H, W]
                class_to_repr[cls] = repr_images
        return class_to_repr

    @staticmethod
    def collate_fn(batch):
        return batch
