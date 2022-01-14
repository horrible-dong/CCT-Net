import csv
import os
import random
import shutil

from PIL import Image

from datasets.classes import *


def make_mini_imagenet(train_csv_path, val_csv_path, test_csv_path, img_path, new_img_path):
    train_label = {}
    val_label = {}
    test_label = {}

    with open(train_csv_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        birth_header = next(csv_reader)
        for row in csv_reader:
            train_label[row[0]] = row[1]

    with open(val_csv_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        birth_header = next(csv_reader)
        for row in csv_reader:
            val_label[row[0]] = row[1]

    with open(test_csv_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        birth_header = next(csv_reader)
        for row in csv_reader:
            test_label[row[0]] = row[1]

    for jpg in os.listdir(img_path):
        if jpg == '.DS_Store':
            print(jpg)
            continue
        path = img_path + '/' + jpg
        im = Image.open(path)
        if jpg in train_label.keys():
            tmp = train_label[jpg]
            temp_path = new_img_path + '/train' + '/' + tmp
            if not os.path.exists(temp_path):
                os.makedirs(temp_path)
            t = temp_path + '/' + jpg
            im.save(t)
            # with open(temp_path, 'wb') as f:
            #     f.write(path)

        elif jpg in val_label.keys():
            tmp = val_label[jpg]
            temp_path = new_img_path + '/val' + '/' + tmp
            if not os.path.exists(temp_path):
                os.makedirs(temp_path)
            t = temp_path + '/' + jpg
            im.save(t)

        elif jpg in test_label.keys():
            tmp = test_label[jpg]
            temp_path = new_img_path + '/test' + '/' + tmp
            if not os.path.exists(temp_path):
                os.makedirs(temp_path)
            t = temp_path + '/' + jpg
            im.save(t)


def rename(root):
    for dir in os.listdir(root):
        os.rename(os.path.join(root, dir), os.path.join(root, mini_imagenet_classes_dict[dir]))


def get_src_and_tgt(root, dataset_path, src_classes, tgt_classes, train_dir=True):
    """
    :param root = dataset/src?-tgt?
    :param dataset_path = original imagefolder-like dataset path e.g. /nfs3-p1/qt/datasets/imagefolder/mini-imagenet

    original imagefolder-like dataset
                                    train
                                        cls1
                                        cls2
                                        ...
                                    (test
                                        cls1
                                        cls2
                                        ...)

    -->
    dataset
        src?-tgt?
            train_src
                    cls1
                    cls2
                    ...
            train_tgt
                    cls1
                    cls2
                    ...
            (test_src
                    cls1
                    cls2
                    ...
            test_tgt
                    cls1
                    cls2
                    ...)
    """
    print(len(src_classes), list(src_classes))
    print(len(tgt_classes), list(tgt_classes))

    dir = "train" if train_dir else "test"
    src = os.path.join(root, dir + "_src")
    tgt = os.path.join(root, dir + "_tgt")

    if not os.path.exists(src):
        os.mkdir(src)
    if not os.path.exists(tgt):
        os.mkdir(tgt)

    for cls in src_classes:
        assert os.path.exists(os.path.join(dataset_path, dir, cls)), print(os.path.join(dataset_path, dir, cls))
        cls_path = os.path.join(dataset_path, dir, cls)
        dst = os.path.join(src, cls)
        print(f"{cls_path} --copy--> {dst}")
        shutil.copytree(cls_path, dst)

    for cls in tgt_classes:
        assert os.path.exists(os.path.join(dataset_path, dir, cls))
        cls_path = os.path.join(dataset_path, dir, cls)
        dst = os.path.join(tgt, cls)
        print(f"{cls_path} --copy--> {dst}")
        shutil.copytree(cls_path, dst)


def split_val_test(mode='src', root='base', val_rate=0.1, test_rate=0.2):
    """
    for imagefolder-like datasets
    if have test set, test_rate = -1

    dataset
        train_src
        train_tgt
        (test_src
        test_tgt)

    -->
    dataset
        train_src
        train_tgt
        val_src
        val_tgt
        test_src
        test_tgt
    """
    assert test_rate > 0 or test_rate == -1

    train_path = os.path.join(root, 'train_' + mode)
    val_path = os.path.join(root, 'val_' + mode)
    os.mkdir(val_path)

    if test_rate > 0:
        test_path = os.path.join(root, 'test_' + mode)
        os.mkdir(test_path)

    for cls in os.listdir(train_path):
        cls_path = os.path.join(train_path, cls)
        imgs = os.listdir(cls_path)
        random.shuffle(imgs)

        val_cls_path = os.path.join(val_path, cls)
        os.mkdir(val_cls_path)
        vaild_num = int(len(imgs) * val_rate)
        for i in range(0, vaild_num):
            img_path = os.path.join(cls_path, imgs[i])
            shutil.move(img_path, val_cls_path)

        if test_rate > 0:
            test_cls_path = os.path.join(test_path, cls)
            os.mkdir(test_cls_path)
            test_num = int(len(imgs) * test_rate)
            for i in range(vaild_num, vaild_num + test_num):
                img_path = os.path.join(cls_path, imgs[i])
                shutil.move(img_path, test_cls_path)

        print(f"class {cls}  val num {vaild_num}" + (f" | test num {test_num}" if test_rate > 0 else ""))


def generate_base_specified_dataset(dst, dataset_path, src_classes: list, tgt_classes: list, val_rate, test_rate):
    """
    for imagefolder-like datasets
    if have test set, test_rate = -1

    :param dst: project/dataset e.g. /nfs/qt/datasets/comparing-GAN/mini-imagenet

    -->
    dataset
        src?-tgt?
            train_src
            train_tgt
        val_src
        val_tgt
        test_src
        test_tgt
    """
    assert (test_rate > 0 or test_rate == -1) and val_rate > 0

    if not os.path.exists(dst):
        os.mkdir(dst)
    base = os.path.join(dst, f'src{len(src_classes)}-tgt{len(tgt_classes)}')
    if not os.path.exists(base):
        os.mkdir(base)

    if test_rate > 0:
        get_src_and_tgt(root=base, dataset_path=dataset_path, src_classes=src_classes, tgt_classes=tgt_classes,
                        train_dir=True)
    elif test_rate == -1:
        get_src_and_tgt(root=base, dataset_path=dataset_path, src_classes=src_classes, tgt_classes=tgt_classes,
                        train_dir=True)
        get_src_and_tgt(root=base, dataset_path=dataset_path, src_classes=src_classes, tgt_classes=tgt_classes,
                        train_dir=False)

    split_val_test(mode='src', root=base, val_rate=val_rate, test_rate=test_rate)
    split_val_test(mode='tgt', root=base, val_rate=val_rate, test_rate=test_rate)
    print()

    print(f"{os.path.join(base, 'val_src')} --move--> {dst}")
    shutil.move(os.path.join(base, "val_src"), dst)
    print(f"{os.path.join(base, 'val_tgt')} --move--> {dst}")
    shutil.move(os.path.join(base, "val_tgt"), dst)
    print(f"{os.path.join(base, 'test_src')} --move--> {dst}")
    shutil.move(os.path.join(base, "test_src"), dst)
    print(f"{os.path.join(base, 'test_tgt')} --move--> {dst}")
    shutil.move(os.path.join(base, "test_tgt"), dst)
    print()


def generate_base_random_dataset(dst, dataset_path, classes: list, num_src_classes: int, num_tgt_classes: int,
                                 val_rate, test_rate):
    """
    for imagefolder-like datasets
    if have test set, test_rate = -1

    :param dst: project/dataset e.g. /nfs/qt/datasets/comparing-GAN/mini-imagenet

    -->
    dataset
        src?-tgt?
            train_src
            train_tgt
        val_src
        val_tgt
        test_src
        test_tgt
    """
    assert (test_rate > 0 or test_rate == -1) and val_rate > 0
    assert len(classes) >= num_src_classes + num_tgt_classes
    if not os.path.exists(dst):
        os.makedirs(dst)
    base = os.path.join(dst, f'src{num_src_classes}-tgt{num_tgt_classes}')
    if not os.path.exists(base):
        os.mkdir(base)
    random.shuffle(classes)
    src_classes = classes[0: num_src_classes]
    tgt_classes = classes[num_src_classes: num_src_classes + num_tgt_classes]

    if test_rate > 0:
        get_src_and_tgt(root=base, dataset_path=dataset_path, src_classes=src_classes, tgt_classes=tgt_classes,
                        train_dir=True)
    elif test_rate == -1:
        get_src_and_tgt(root=base, dataset_path=dataset_path, src_classes=src_classes, tgt_classes=tgt_classes,
                        train_dir=True)
        get_src_and_tgt(root=base, dataset_path=dataset_path, src_classes=src_classes, tgt_classes=tgt_classes,
                        train_dir=False)

    split_val_test(mode='src', root=base, val_rate=val_rate, test_rate=test_rate)
    split_val_test(mode='tgt', root=base, val_rate=val_rate, test_rate=test_rate)
    print()

    print(f"{os.path.join(base, 'val_src')} --move--> {dst}")
    shutil.move(os.path.join(base, "val_src"), dst)
    print(f"{os.path.join(base, 'val_tgt')} --move--> {dst}")
    shutil.move(os.path.join(base, "val_tgt"), dst)
    print(f"{os.path.join(base, 'test_src')} --move--> {dst}")
    shutil.move(os.path.join(base, "test_src"), dst)
    print(f"{os.path.join(base, 'test_tgt')} --move--> {dst}")
    shutil.move(os.path.join(base, "test_tgt"), dst)
    print()


def generate_few_shot_sub_datasets(root, num_src_classes, num_tgt_classes, fs: int, percent=True):
    """
    :param root: project/dataset e.g. /nfs/qt/datasets/comparing-GAN/mini-imagenet
    :param fs: number or percent of samples for few-shot task

    -->
    dataset
        src?-tgt?
                train_src
                train_tgt
        ?-shot-src?-tgt?
                train_few_shot_tgt
                train_src
                train_tgt
        val_src
        val_tgt
        test_src
        test_tgt
    """
    assert fs >= 1

    base = os.path.join(root, f'src{num_src_classes}-tgt{num_tgt_classes}')
    assert os.path.exists(base)

    if percent:
        dst = os.path.join(root, f'{fs}p-shot-src{num_src_classes}-tgt{num_tgt_classes}')
        fs /= 100
    else:
        dst = os.path.join(root, f'{fs}-shot-src{num_src_classes}-tgt{num_tgt_classes}')

    print(f'{base} --copy--> {dst}\n')
    shutil.copytree(base, dst)

    train_tgt = os.path.join(dst, 'train_tgt')
    train_fs_tgt = os.path.join(dst, "train_few_shot_tgt")
    os.mkdir(train_fs_tgt)

    for cls in os.listdir(train_tgt):
        os.mkdir(os.path.join(train_fs_tgt, cls))
        cls_path = os.path.join(train_tgt, cls)
        dst_cls_path = os.path.join(train_fs_tgt, cls)
        list_img = os.listdir(cls_path)
        num_fs = int(fs * len(list_img)) if percent else fs
        fs_imgs = random.sample(list_img, num_fs)
        print(f"{cls_path} --move-{num_fs}--> {os.path.join(train_fs_tgt, cls)}")
        for img in fs_imgs:
            shutil.move(os.path.join(cls_path, img), os.path.join(dst_cls_path, img))
    print()


def make_test_dir(test_dir, dataset_path, classes):
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    for cls in classes:
        if os.path.exists(os.path.join(dataset_path, "train", cls)):
            cls_path = os.path.join(dataset_path, "train", cls)
        elif os.path.exists(os.path.join(dataset_path, "val", cls)):
            cls_path = os.path.join(dataset_path, "val", cls)
        elif os.path.exists(os.path.join(dataset_path, "test", cls)):
            cls_path = os.path.join(dataset_path, "test", cls)

        dst = os.path.join(test_dir, cls)
        print(f"{cls_path} --copy--> {dst}")
        shutil.copytree(cls_path, dst)


def example():
    dataset = 'mini-imagenet'
    classes = mini_imagenet_classes
    num_src_classes, num_tgt_classes = 50, 50

    generate_base_random_dataset(dst=f'./datasets/comparing-GAN/{dataset}',
                                 dataset_path=f'./datasets/imagefolder/{dataset}',
                                 classes=classes,
                                 num_src_classes=num_src_classes,
                                 num_tgt_classes=num_tgt_classes,
                                 val_rate=0.1,
                                 test_rate=0.2)

    root = f'./datasets/comparing-GAN/{dataset}'
    percent = False
    generate_few_shot_sub_datasets(root=root,
                                   num_src_classes=num_src_classes,
                                   num_tgt_classes=num_tgt_classes,
                                   fs=5,
                                   percent=percent)

    generate_few_shot_sub_datasets(root=root,
                                   num_src_classes=num_src_classes,
                                   num_tgt_classes=num_tgt_classes,
                                   fs=20,
                                   percent=percent)
