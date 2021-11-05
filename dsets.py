# FILE AND SYSTEM
import glob
import os
import os.path as osp
import sys

# MATH AND ARRAY OP
import SimpleITK as sitk
import numpy as np
import pandas as pd
import ast
import random
import scipy.ndimage as ndi

# CACHE
from diskcache import FanoutCache
# from util.disk import getCache
from boltons.cacheutils import cachedproperty
import functools

# TORCH
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import model

#DICOM_DIR = '/home/scarpma/db/dicom_files_MARTINO_NON_MODIFICARE/'
DICOM_DIR = '../voxel2meshRuns/'
CACHE_DIR = '../voxel2meshData/.cache_data'

HU_BOTTOM_LIM, HU_TOP_LIM = -200, 800
THRESHOLD = 0.5
ROI_IS_mm = 215 # INFERIOR - SUPERIOR
ROI_LR_mm = 144 # LEFT - RIGHT
ROI_PA_mm = 180 # POSTERIOR - ANTERIOR
ROI_IS_mm_p = 35  # above arch
ROI_IS_mm_n = 180 # below arch
FINAL_SIZE = [384,320,256]
# px spacing : array([ 0.55989583 , 0.5625 , 0.5625 ])


#raw_cache = getCache(osp.join(DICOM_DIR, '.raw'))
raw_cache = FanoutCache(osp.join(CACHE_DIR, 'cache'),
    #disk=GzipDisk,
    shards=64,
    timeout=1,
    size_limit=3e11,
    # disk_min_file_size=2**20,
)



@functools.lru_cache(1)
def getPatientArchive(root_dir=DICOM_DIR):
    ## READ PATIENT INFO CSV FILE
    archive_file = osp.join(root_dir,'patients.csv')
    archive = pd.read_csv(
        archive_file,
        index_col=0,
        converters={
            "size": ast.literal_eval,
            "vxSize_xyz": ast.literal_eval,
            "origin_xyz": ast.literal_eval,
        }
    )
    return archive



def buildPatientArchive(root_dir=DICOM_DIR, new_archive_bool=False):
    archive_file = osp.join(root_dir,'patients.csv')
    patient_dir_list = glob.glob(osp.join(root_dir, '*/[!ROI_arrays]'))
    if not osp.exists(archive_file): new_archive_bool = True
    archive_cols = ['name',
        'series_uid','size','vxSize_xyz',
        'origin_xyz','available_labelmaps',
    ]
    if new_archive_bool:
        archive = pd.DataFrame(columns=archive_cols)
    else:
        archive = getPatientArchive(root_dir)

        ## LOOK FOR NEW UN-ARCHIVED PATIENTS
        patient_dir_list = [pat for pat in patient_dir_list if
            not pat_name_from_dir(pat) in archive['name'].values
        ]
        if len(patient_dir_list) >= 1:
            print("new patients to add:")
            print(patient_dir_list)

    ## ADD NEW PATIENTS TO ARCHIVE
    for ii, patient_dir in enumerate(patient_dir_list):
        ct = Ct(patient_dir)
        ct.hu_a # reads ct scan to get metadata
        patient_entries = [
            ct.patient_name,
            ct.series_uid,
            ct.size,
            ct.vxSize_xyz,
            ct.origin_xyz,
            int(ct.int_labelmap.max()),
        ]
        del ct
        archive.loc[archive.index.max()+1] = patient_entries
        archive.to_csv(archive_file)

    ## REORDER ARCHIVE AND REMOVE DUPLICATES
    archive = archive.sort_values('series_uid')
    archive = archive.reset_index(drop=True)
    print(archive[['name','size','vxSize_xyz','available_labelmaps']])
    archive.to_csv(archive_file)



def pat_name_from_dir(patient_dir):
    return osp.basename(osp.normpath(patient_dir))



@functools.lru_cache(1, typed=True) # temporary cache
def getCt(patient_name):
    return Ct(osp.join(DICOM_DIR, patient_name))


@functools.lru_cache(4, typed=True) # temporary cache
def getResampledRoi(patient_name):
    ROI = np.load(
        osp.join(DICOM_DIR, 'ROI_arrays', f'{patient_name}.npy'),
        allow_pickle=True,
        )[()]
    hu_a = ROI['hu_a']
    int_labelmap = ROI['int_labelmap']
    resample_factor = ROI['resample_factor']
    hu_a_roi = ndi.zoom(
        hu_a,
        resample_factor,
        order=1,
        mode='constant',
        cval=HU_BOTTOM_LIM)
    int_labelmap_roi = ndi.zoom(
        int_labelmap.astype(np.float32),
        resample_factor,
        order=1,
        mode='constant',
        cval=0) > THRESHOLD
    return hu_a_roi, int_labelmap_roi


def getCtSlice(patient_name, slice_idx, proj='axial'):

    ROI = getResampledRoi(patient_name)[0]

    if proj == 'sagittal':
        ROI = ROI[::-1]
        ROI = ROI.transpose((2,0,1))
    elif proj == 'coronal':
        ROI = ROI[::-1]
        ROI = ROI.transpose((1,0,2))
    else:
        pass

    return ROI[slice_idx].copy()



def getLabelmapSlice(patient_name, slice_idx, proj='axial'):
    ROI = getResampledRoi(patient_name)[1]

    if proj == 'sagittal':
        ROI = ROI[::-1]
        ROI = ROI.transpose((2,0,1))
    elif proj == 'coronal':
        ROI = ROI[::-1]
        ROI = ROI.transpose((1,0,2))
    else:
        pass

    return ROI[slice_idx].copy()



class Ct(object):
    def __init__(self, patient_dir):
        self.patient_dir = patient_dir
        self.patient_name = pat_name_from_dir(patient_dir)
        self.reader = sitk.ImageSeriesReader()
        archive = getPatientArchive().set_index('name', drop=False)
        self.info = archive.loc[self.patient_name]
        self.arch_z_coord = float(self.info['arch_z_coord'])
        del archive

    @cachedproperty
    def origin_xyz(self):
        metadata = self.getMetadata(self.dicom_files[0])
        return metadata['0020|0032'].split('\\')

    @cachedproperty
    def dcm_dir(self):
        search_path = self.patient_dir
        while len(glob.glob(osp.join(search_path, '*.dcm'))) < 100 :
            search_path = glob.glob(osp.join(search_path, '*/'))
            assert len(search_path) == 1, print('multiple dir', search_path)
            search_path = search_path[0]
        return search_path

    @cachedproperty
    def series_uid(self):
        series_uid = self.reader.GetGDCMSeriesIDs(self.dcm_dir)
        assert len(series_uid) == 1,  print(len(series_uid))
        return series_uid[0]

    @cachedproperty
    def dicom_files(self):

        ## # load all files in the right order
        ## files = self.reader.GetGDCMSeriesFileNames(self.dcm_dir)
        ## # remove all files not ending with '.dcm'
        ## files = [fn for fn in files if '.dcm' in fn[-4:]]

        pos_tag = '0020|0032'
        positions = []
        dicom_files = glob.glob(osp.join(self.dcm_dir, '*.dcm'))
        for slice_idx in range(len(dicom_files)):
            positions.append(self.getMetadata(
                dicom_files[slice_idx]
                                          )[pos_tag].split('\\'))
        positions = np.array(positions).astype(np.float32)
        idxs = np.argsort(positions[:,2])
        positions = positions[idxs]
        dicom_files_ordered = [dicom_files[i] for i in idxs]

        if np.diff(positions[:,2]).ptp() > 0.05:
            print('detected slice distance mismatch in {} of {} mm'.format(
                self.patient_name, np.diff(positions[:,2]).ptp()))

        return dicom_files_ordered

    @cachedproperty
    def hu_a(self):
        print("Reading :", self.patient_name)
        self.reader.SetFileNames(self.dicom_files)
        ct_dicom = self.reader.Execute()
        self.vxSize_xyz = ct_dicom.GetSpacing()

        hu_a = np.array(sitk.GetArrayFromImage(ct_dicom), dtype=np.float32)
        # .GetSize counts file, not actual shape
        self.size = hu_a.shape # ct_dicom.GetSize()
        return hu_a

    @cachedproperty
    def int_labelmap(self):
        # ATTENZIONE, INT_LABELMAP CONTIENE ANCHE ATRIO E VENTRICOLO
        import nrrd
        path = osp.join(self.patient_dir, 'seg.nrrd')
        seg, seg_header = nrrd.read(path)
        seg = seg.T.copy()
        assert self.hu_a.shape == seg.shape, '{}, {}'.format(
            self.hu_a.shape, seg.shape)
        return seg

    @cachedproperty
    def bool_labelmap(self):
        bool_labelmap = self.int_labelmap == 1
        return bool_labelmap

    @cachedproperty
    def roi_bounds(self):
        vx_x, vx_y, vx_z = self.info.vxSize_xyz
        zorigin = float(self.origin_xyz[2])
        di_temp = round((ROI_IS_mm/2) / vx_z) # temporary displacement
        di_p = round(ROI_IS_mm_p / vx_z)  # positive displacement
        di_n = round(ROI_IS_mm_n / vx_z) # negative displacement
        arch_index = round((self.arch_z_coord-zorigin)/vx_z)
        assert 0 < arch_index < self.hu_a.shape[0]
        maxi = arch_index
        mini = max([maxi - di_temp*2,0])

        idxs = np.where(self.bool_labelmap[mini:maxi].astype(int)==1)
        maxi = arch_index + di_p
        mini = max([arch_index - di_n,0])
        maxr = idxs[1].max()
        minr = idxs[1].min()
        maxc = idxs[2].max()
        minc = idxs[2].min()
        cr, cc = round(0.5*(maxr+minr)), round(0.5*(maxc+minc))
        dr = round((ROI_PA_mm/2) / vx_y)
        dc = round((ROI_LR_mm/2) / vx_x)
        minr, maxr = max([cr - dr,0]), min([cr + dr,self.hu_a.shape[1]-1])
        minc, maxc = max([cc - dc,0]), min([cc + dc,self.hu_a.shape[2]-1])

        return mini,maxi,minr,maxr,minc,maxc

    @cachedproperty
    def roi(self):
        mini,maxi,minr,maxr,minc,maxc = self.roi_bounds
        hu_a_roi = self.hu_a[mini:maxi,minr:maxr,minc:maxc]
        int_labelmap_roi = self.bool_labelmap[mini:maxi,minr:maxr,minc:maxc].astype(int)
        return hu_a_roi, int_labelmap_roi

    @cachedproperty
    def resample_factor(self, size=FINAL_SIZE):
        return np.array(size) / np.array(self.roi[0].shape)

    def resampled_roi(self, order=1, size=FINAL_SIZE):
        resample_factor = np.array(size) / np.array(self.roi[0].shape)
        vx_x, vx_y, vx_z = self.info.vxSize_xyz
        new_vx_z, new_vx_x, new_vx_y = np.array([vx_z, vx_x, vx_y]) / resample_factor
        print(new_vx_z, new_vx_x, new_vx_y)
        hu_a_roi = ndi.zoom(
            self.roi[0],
            resample_factor,
            order=order,
            mode='constant',
            cval=HU_BOTTOM_LIM)
        int_labelmap_roi = ndi.zoom(
            self.roi[1].astype(np.float32),
            resample_factor,
            order=order,
            mode='constant',
            cval=0)
        int_labelmap_roi = int_labelmap_roi > THRESHOLD
        int_labelmap_roi = int_labelmap_roi.astype(np.int32)
        print(hu_a_roi.shape)
        return hu_a_roi, int_labelmap_roi

    def getMetadata(self, dcmFilename):
        reader = sitk.ImageFileReader()
        reader.SetFileName(dcmFilename)
        reader.ReadImageInformation()
        reader.GetLoadPrivateTags()
        metadata = {}
        for k in reader.GetMetaDataKeys():
            metadata[k] = reader.GetMetaData(k)

        return metadata


def rgb_overlay(bg, mask):
    ''' return an rgb image array with highlighted labelmap
        and its countours. Useful to check if labelmap and
        scan are correctely registered '''

    from skimage.color import label2rgb
    from skimage import segmentation

    if type(bg) == type(torch.Tensor()):
        bg = bg.numpy().astype(float)
    else: bg = bg.astype(float)
    if type(mask) == type(torch.Tensor()):
        mask = mask.numpy().astype(float)
    else: mask = mask.astype(float)

    bg = (bg - bg.min()) / bg.ptp()

    # bg stands for background image (the ct scan)
    color_filled = label2rgb(mask, bg, bg_label=0, colors=[(1,0,0)], alpha=0.2)
    rgb_overlay = segmentation.mark_boundaries(
            color_filled, mask,color=(1,0,0), mode='thick'
    )
    return rgb_overlay



def check_volume_alinment(bg, masks=None, num_imgs=36):
    start = 0
    stop = len(bg) - 1
    img_idxs = np.linspace(start,stop,num_imgs,dtype=np.int)
    if isinstance(bg, Segmentation2DDataset):
        img_list = [bg[idx][0][bg.contextSlices_count].numpy() for idx in img_idxs]
    else:
        img_list =[bg[idx] for idx in img_idxs]
    if masks is None:
        mask_list = [bg[idx][1][0].numpy().astype(int) for idx in img_idxs]
    else:
        mask_list = [masks[idx].astype(int) for idx in img_idxs]

    overlay_list = [rgb_overlay(img, mask) for img, mask in zip(img_list, mask_list)]
    fig = plot_mosaic(overlay_list)
    return fig



def plot_mosaic(img_list):
    import matplotlib.pyplot as plt
    max_rows = int(np.sqrt(len(img_list)))
    max_cols = int(np.ceil(len(img_list)/max_rows))
    img_shape = img_list[0].shape
    fig_size = (14*(max_cols*img_shape[1])/(max_rows*img_shape[0]),14)
    fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=fig_size)
    for idx, image in enumerate(img_list):
        row = idx // max_cols
        col = idx % max_cols
        axes[row, col].axis("off")
        aspect = 1
        axes[row, col].imshow(image, aspect=aspect)
    plt.subplots_adjust(wspace=.05, hspace=.05)
    return fig



class SegmentationDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 patient_name=None,
            ):
        self.archive = getPatientArchive().set_index('name', drop=False)

        self.n_slices = FINAL_SIZE[0]
        self.n_rows = FINAL_SIZE[1]
        self.n_cols = FINAL_SIZE[2]

        ## BUILD PATIENT LIST AS A SUBSET OF PATIENT NAMES FROM ARCHIVE
        if patient_name:
            self.patients_list = [patient_name]
            assert patient_name in self.archive['name'].values
        else:
            self.patients_list = list(self.archive['name'].values)

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.patients_list = self.patients_list[::val_stride]
            assert self.patients_list
        elif val_stride > 0:
            del self.patients_list[::val_stride]
            assert self.patients_list

        ## BUILD SAMPLE LIST, IE A LIST OF TUPLES (PATIENT_NAME, SLICE_IDX)
        ## FOR ALL SLICES FROM ALL PATIENT IN PATIENT_LIST
        self.sample_list = []
        for patient_name in self.patients_list:
            self.sample_list += [patient_name]

        print("{!r}: {} {} series".format(
            self,
            len(self.patients_list),
            {None: 'general', True: 'validation', False: 'training'}[isValSet_bool],
        ))




    def shuffleSamples(self):
        random.shuffle(self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        patient_name = self.sample_list[idx % len(self.sample_list)]
        return self.getitem_TrainingSample(patient_name)


    def getitem_TrainingSample(self, patient_name):

        ## this function call should be cached in disk
        x, y = getResampledRoi(patient_name)
        shape = torch.tensor(y.shape)[None].float()
        x = torch.from_numpy(x[0]).unsqueeze(-1)
        y = torch.from_numpy(y[0]).unsqueeze(-1)

        ## WINDOWING
        x.clamp_(HU_BOTTOM_LIM, HU_TOP_LIM)

        ## data augmentation to be implemented

        ## sample outer surface
        surface_points_normalized_all = []
        y_outer = sample_outer_surface_in_voxel((y == 1).long())
        surface_points = torch.nonzero(y_outer)
        surface_points = torch.flip(
            surface_points, dims=[1]).float()  # convert z,y,x -> x, y, z
        surface_points_normalized = normalize_vertices(surface_points, shape)

        perm = torch.randperm(len(surface_points_normalized))
        point_count = 3000
        surface_points_normalized_all += [
            surface_points_normalized[
                perm[:np.min([len(perm), point_count])]].cuda()
        ]  # randomly pick 3000 points

        return {
            "x": x,
            "y_voxels": y,
            "surface_points": surface_points_normalized,
        }



def sample_outer_surface_in_voxel(volume):

    # grows a layer of voxels around the volume, like an outer voxelized surface.
    # returns the volume array with ones on these "surface" voxels.

    # inner surface
    # a = F.max_pool3d(-volume[None,None].float(), kernel_size=(3,1,1), stride=1, padding=(1, 0, 0))[0]
    # b = F.max_pool3d(-volume[None,None].float(), kernel_size=(1,3, 1), stride=1, padding=(0, 1, 0))[0]
    # c = F.max_pool3d(-volume[None,None].float(), kernel_size=(1,1,3), stride=1, padding=(0, 0, 1))[0]
    # border, _ = torch.max(torch.cat([a,b,c],dim=0),dim=0)
    # surface = border + volume.float()

    # outer surface
    a = F.max_pool3d(volume[None, None].float(),
                     kernel_size=(3, 1, 1),
                     stride=1,
                     padding=(1, 0, 0))[0]
    b = F.max_pool3d(volume[None, None].float(),
                     kernel_size=(1, 3, 1),
                     stride=1,
                     padding=(0, 1, 0))[0]
    c = F.max_pool3d(volume[None, None].float(),
                     kernel_size=(1, 1, 3),
                     stride=1,
                     padding=(0, 0, 1))[0]
    border, _ = torch.max(torch.cat([a, b, c], dim=0), dim=0)
    surface = border - volume.float()
    return surface.long()



def normalize_vertices(vertices, shape):
    assert len(vertices.shape) == 2 and len(
        shape.shape) == 2, "Inputs must be 2 dim"
    assert shape.shape[0] == 1, "first dim of shape should be length 1"

    print(shape)
    print(torch.max(vertices))
    return 2 * (vertices / (torch.max(shape) - 1) - 0.5)



def clean_border_pixels(image, gap):
    '''
    :param image:
    :param gap:
    :return:
    '''
    assert len(image.shape) == 3, "input should be 3 dim"

    D, H, W = image.shape
    y_ = image.clone()
    y_[:gap] = 0
    y_[:, :gap] = 0
    y_[:, :, :gap] = 0
    y_[D - gap:] = 0
    y_[:, H - gap] = 0
    y_[:, :, W - gap] = 0

    return y_



'''
@functools.lru_cache(1)
def getTotSlices():
    count = 0
    archive = getPatientArchive()
    for row in archive.iloc():
        count += row['size'][-1]

    return count



def prepCache():
    archive = getPatientArchive()
    patient_names = archive['name'].values
    tot_slices = getTotSlices()
    perc = 0.
    for ii, patient_name in enumerate(patient_names) :
        perc += archive.iloc[ii]['size'][-1]/tot_slices
        print('\n{:>2}) patient {}, {:.1f} % slices done :\n'.format(
            ii, patient_name, perc*100))
        hu_a, bool_labelmap = getCtArray(patient_name)
        print(hu_a.shape, bool_labelmap.shape)
'''


if __name__ == '__main__':

    ## buildPatientArchive()

    ##if True:
    ##    #raw_cache.clear()
    ##    print('')
    ##    print('-'*20)
    ##    print('save single ROIs')
    ##    print('-'*20)
    ##    print('')

    ##    patients_list = list(getPatientArchive()['name'].values)
    ##    for patient in patients_list:
    ##        print(patient)
    ##        save_path = osp.join('data', 'ROI_arrays', f'{patient}.npy')
    ##        if osp.exists(save_path):
    ##            print('Already exists. Skipping ...')
    ##            continue
    ##        ct = getCt(patient)
    ##        bg, mask = ct.roi
    ##        print(bg.shape)
    ##        array_tosave = {
    ##            'hu_a'            : bg,
    ##            'int_labelmap'    : mask.astype(bool),
    ##            'roi_bounds'      : ct.roi_bounds,
    ##            'resample_factor' : ct.resample_factor,
    ##            'vxSize_xyz'      : ct.info.vxSize_xyz,
    ##            'origin_xyz'      : ct.origin_xyz,
    ##        }
    ##        np.save(save_path, array_tosave)

    ds = SegmentationDataset()
    a = ds[0]
    print(a)
    print(torch.max(a['surface_points']))


## import scipy.ndimage as nd
## import pyvista as pv
##
## def marching_cubes(array):
##     array = array.squeeze()
##     assert len(array.shape)==3
##     a = pv.wrap(array)
##     contour = a.contour(
##             isosurfaces=1,
##             rng=(THRESHOLD,1),
##             method='marching_cubes',
##             progress_bar=True,
##     )
##     return contour
##
## # from resampled roi to
## surf = marching_cubes(mask.transpose((2,1,0)))
## mini,maxi,minr,maxr,minc,maxc = ct.roi_bounds
## origin = np.array([minc, minr, mini])
## surf.points = (surf.points/ct.resample_factor[np.array([2,1,0])]) + origin
## surf.points = (ct.info.vxSize_xyz*surf.points) + np.array(ct.origin_xyz, dtype=float)
