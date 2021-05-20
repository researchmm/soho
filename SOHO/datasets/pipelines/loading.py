import os.path as osp
import warnings
import zipfile

import numpy as np

import commons
from ..registry import PIPELINES


# import pycocotools.mask as maskUtils


class ZipReader(object):
    def __init__(self):
        super(ZipReader, self).__init__()
        self.id_context = dict()

    def read(self, zip_file, image_name, pid=0):
        key_name = zip_file + '_' + str(pid)
        if key_name in self.id_context:
            with self.id_context[key_name].open(image_name) as f:
                tmp = f.read()
            return tmp
        else:
            file_handle = zipfile.ZipFile(zip_file, 'r', zipfile.ZIP_LZMA)
            self.id_context[key_name] = file_handle
            return self.id_context[key_name].read(image_name)


@PIPELINES.register_module
class LoadImageFromZip(object):
    def __init__(self, to_float32=False):
        self.to_float32 = to_float32
        self.zipreader = ZipReader()

    def __call__(self, results):
        if results['img_prefix'].endswith("/"):
            zip_path = results['img_prefix'][:-1] + ".zip"
            img_path = results['img_prefix'].split("/")[-2]
        else:
            zip_path = results['img_prefix'] + ".zip"
            img_path = results['img_prefix'].split("/")[-1]
        if "images" in img_path:
            filename = results['img_info']['filename']
        else:

            filename = osp.join(img_path, results['img_info']['filename'])
        tmp = self.zipreader.read(zip_path, filename)
        img = commons.imfrombytes(tmp)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


@PIPELINES.register_module
class LoadImageFromZipNV(object):
    def __init__(self, to_float32=False):
        self.to_float32 = to_float32
        self.zipreader = ZipReader()

    def __call__(self, results):
        zip_path = results['img_prefix'][:-1] + ".zip"
        img_path = results['img_prefix'].split("/")[-2]
        if "images" in img_path:
            filename = results['img_info']['filename']
        else:

            filename = osp.join(img_path, results['img_info']['filename'])
        img = self.zipreader.read(zip_path, filename)

        # if self.to_float32:
        #     img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


@PIPELINES.register_module
class LoadImageFromZipVCR(object):
    def __init__(self, to_float32=False):
        self.to_float32 = to_float32
        self.zipreader = ZipReader()

    def __call__(self, results):
        zip_path = results['img_prefix'][:-1] + ".zip"
        img_path = results['img_prefix'].split("/")[-2]
        if "images" in img_path:
            filename = results['img_info']['filename']
        else:

            filename = osp.join(img_path, results['img_info']['filename'])
        print("file_name", filename)
        tmp = self.zipreader.read(zip_path, filename)
        img = commons.imfrombytes(tmp)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


@PIPELINES.register_module
class LoadImageFromFile(object):

    def __init__(self, to_float32=False):
        self.to_float32 = to_float32

    def __call__(self, results):
        filename = osp.join(results['img_prefix'],
                            results['img_info']['filename'])
        img = commons.imread(filename)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)


@PIPELINES.register_module
class LoadPairImageFromZipFile(object):
    def __init__(self, to_float32=False):
        self.to_float32 = to_float32
        self.zipreader = ZipReader()

    def __call__(self, results):
        zip_path = results['img_prefix'][:-1] + ".zip"
        img_path = results['img_prefix'].split("/")[-2]

        filename1 = osp.join(results['img_prefix'],
                             results['img_info']['filename1'])

        tmp1 = self.zipreader.read(zip_path, filename1)
        img1 = commons.imfrombytes(tmp1)

        filename2 = osp.join(results['img_prefix'],
                             results['img_info']['filename2'])

        tmp2 = self.zipreader.read(zip_path, filename2)
        img2 = commons.imfrombytes(tmp2)

        if self.to_float32:
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)

        results['filename1'] = filename1
        results['img1'] = img1
        results['img1_shape'] = img1.shape
        results['ori1_shape'] = img1.shape

        results['filename2'] = filename2
        results['img2'] = img2
        results['img2_shape'] = img2.shape
        results['ori2_shape'] = img2.shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)

@PIPELINES.register_module
class LoadPairImagesFromFile(object):

    def __init__(self, to_float32=False):
        self.to_float32 = to_float32

    def __call__(self, results):
        filename1 = osp.join(results['img_prefix'],
                             results['img_info']['filename1'])
        img1 = commons.imread(filename1)

        filename2 = osp.join(results['img_prefix'],
                             results['img_info']['filename2'])
        img2 = commons.imread(filename2)

        if self.to_float32:
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)

        results['filename1'] = filename1
        results['img1'] = img1
        results['img1_shape'] = img1.shape
        results['ori1_shape'] = img1.shape

        results['filename2'] = filename2
        results['img2'] = img2
        results['img2_shape'] = img2.shape
        results['ori2_shape'] = img2.shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)

