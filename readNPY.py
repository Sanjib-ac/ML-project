import numpy as np
from numba import jit
# import matplotlib.pyplot as plt
# import pandas as pd
# # import cudf
# import numba
# import cupy as cp
# from timeit import default_timer as timer

from nptdms import TdmsWriter, RootObject, GroupObject, ChannelObject


# # @jit
# def func_a(np_array):
#     return

def return_i_q(nd_ary):
    return nd_ary.real, nd_ary.imag


def gain(ndarray, gn, f16):
    return np.divide(np.multiply(ndarray, gn), f16)


@jit
def iq_vst(ndarray_r, ndarray_i):
    lst = []
    # result = np.empty(l, dtype='h')
    for i in range(len(ndarray_r)):
        lst.extend([ndarray_r[i], ndarray_i[i]])
    return np.array(lst)

# # loading the npy data
# data2d = np.load('dataset_hf_radio.npy')
# n_real, n_img = return_i_q(data2d.flatten())  # convert 2D to 1D,  and get i and  q  two ndarrays

# print(n_real.shape, n_img.shape)
# print(type(n_real), n_img.shape)


# mxr = np.amax(n_real)
# mnr = np.amin(n_real)
# mxi = np.amin(n_img)
# mni = np.amax(n_img)
# print(f'min_Real:{mnr}, maxReal:{mxr}, min_img:{mni}, max_img:{mxi}')
# g = 2.4e3
# fctr = 2 * (max(mxr, mnr, mxi, mnr) - min(mxr, mnr, mxi, mnr)) / 32767
# n_real_scld = gain(n_real, g, fctr)
# n_img_scld = gain(n_img, g, fctr)
#
# print(f'fctr: {fctr}, \n, n_real:{n_real}, \n n_real_scld:{n_real_scld}'
#       f'\n max :{max(n_real_scld)}, min: {min(n_real_scld)}'
#       f'\n max :{max(n_img_scld)}, min: {min(n_img_scld)}')

# iqdata_np = iq_vst(n_real_scld, n_img_scld)

# print(iqdata_np.shape)

def iq_interleaved(nparra):
    n_real, n_img = return_i_q(nparra.flatten())  # convert 2D to 1D,  and get i and  q  two ndarrays
    print(n_real.shape, n_img.shape)
    mxr = np.amax(n_real)
    mnr = np.amin(n_real)
    mxi = np.amin(n_img)
    mni = np.amax(n_img)
    print(f'min_Real:{mnr}, maxReal:{mxr}, min_img:{mni}, max_img:{mxi}')
    g = 2.4e3
    fctr = 2 * (max(mxr, mnr, mxi, mnr) - min(mxr, mnr, mxi, mnr)) / 32767
    n_real_scld = gain(n_real, g, fctr)
    n_img_scld = gain(n_img, g, fctr)

    print(f'fctr: {fctr}, \n, n_real:{n_real}, \n n_real_scld:{n_real_scld}'
          f'\n max :{max(n_real_scld)}, min: {min(n_real_scld)}'
          f'\n max :{max(n_img_scld)}, min: {min(n_img_scld)}')

    iqdata_np = iq_vst(n_real_scld, n_img_scld)
    print(iqdata_np.shape)
    return iqdata_np


def write_tdms(data, gan=1):
    root_object = RootObject(properties={
        "prop1": "Streaming_vst"
    })
    group_object = GroupObject("vst", properties={
        "gain": gan,
        "offset": 0.0,
    })
    channel_object = ChannelObject("1", "channel_1", data, properties={})

    with TdmsWriter("demo_file.tdms") as tdms_writer:
        # Write first segment
        tdms_writer.write_segment([
            root_object,
            group_object,
            channel_object])


if __name__ == "__main__":
    # loading the npy data
    data2d = np.load('dataset_hf_radio.npy')
    final_np = iq_interleaved(data2d)
    write_tdms(final_np)


# scaling_factor = (np.amax(n_real.to_list) - np.amin(n_real) / np.amax(n_img) - np.amin(n_img))
# print(x, np.argmax(n_real))
# print(y, np.argmin(n_real))
# print(n_real[np.argmax(n_real)])
# print(n_real[np.argmin(n_real)])

# print(data2d[0][1:2], n_real[:], n_img[1:3])
# print(np.iscomplex(data))
# printing initial arrays
# print("initial array", str(data))
# print('Shape of the array: ', data.shape)
#
# # Multiplying arrays
# # data1d = data.flatten()
# # print('Shape of the array: ', data1d.shape)
# # print("1D array array", str(data))
# print('Type: ', type(data[1][1]))

# plt.plot(data[1][1:2048])
# plt.show()
# import torch
#
# data_torch = torch.from_numpy(data2d)
# print(data_torch)
