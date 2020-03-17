from numba import jit, cuda, prange
import numpy as np
import copy
from time import time
import cv2
import math


def np_normalize(image: np.array, mean: np.array, std: np.array) -> np.array:
    return (image - mean) / std


def for_normalize(image: np.array, mean: np.array, std: np.array) -> np.array:
    h, w, c = image.shape
    for i in range(h):
        for j in range(w):
            for k in range(c):
                image[i, j, k] = (image[i, j, k] - mean[k]) / std[k]
    return image


@jit(nopython=True, parallel=True)
def jit_for_normalize(image: np.array, mean: np.array, std: np.array) -> np.array:
    h, w, c = image.shape
    for i in prange(h):
        for j in prange(w):
            for k in prange(c):
                image[i, j, k] = (image[i, j, k] - mean[k]) / std[k]
    return image


@jit(nopython=True, parallel=True)
def jit_np_normalize(image: np.array, mean: np.array, std: np.array) -> np.array:
    return (image - mean) / std


@cuda.jit
def cuda_normalize(image: np.array, mean: np.array, std: np.array) -> np.array:
    tx = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    ty = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    
    if tx < image.shape[1] and ty < image.shape[0]:
        for i in range(image.shape[2]):
            image[ty, tx, i] = (image[ty, tx, i] - mean[i]) / std[i] 


def main():
    image = cv2.imread("600044.jpg").astype(np.float32)
    mean = np.array([103.530, 116.280, 123.675])  # Imagenet pixel value in BGR order
    std = np.array([1.0, 1.0, 1.0])

    image1 = copy.deepcopy(image)  
    t1 = time()
    result = for_normalize(image1, mean, std)
    t2 = time()
    print(f"result for normalize {result[1, 0, :]}")    
    del image1
    del result

    image2 = copy.deepcopy(image)
    t3 = time()
    result = np_normalize(image, mean, std)
    t4 = time()
    print(f"result np normalize {result[1, 0, :]}")
    del image2
    del result

    image3 = copy.deepcopy(image)
    t5 = time()
    result = jit_np_normalize(image3, mean, std)
    t6 = time()
    print(f"result jit_np_normalize first {result[1, 0, :]}")
    del image3
    del result

    image4 = copy.deepcopy(image)
    t7 = time()
    result = jit_np_normalize(image4, mean, std)
    t8 = time()
    print(f"result jit_np_normalize second {result[1, 0, :]}")
    del image4
    del result

    # jit normalize
    image5 = copy.deepcopy(image)
    t9 = time()
    result = jit_for_normalize(image5, mean, std)
    t10 = time()
    print(f"result jit_for_normalize first {result[1, 0, :]}")
    del image5
    del result

    image6 = copy.deepcopy(image)
    t11 = time()
    result = jit_for_normalize(image6, mean, std)
    t12 = time()
    print(f"result jit_for_normalize second {result[1, 0, :]}")
    del image6
    del result

    # cuda.jit
    image7 = copy.deepcopy(image)
    # 256 thread per block. Don't exceed 1024.
    threadsperblock = (16, 16)
    blockspergrid_y = math.ceil(image7.shape[0] / threadsperblock[0])
    blockspergrid_x = math.ceil(image7.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    t13 = time()
    cuda_normalize[blockspergrid, threadsperblock](image7, mean, std) 
    t14  = time()
    print(f"result cuda_normalize first {image7[1, 0, :]}") 
    del image7

    # cuda.jit
    image8 = copy.deepcopy(image)
    t15 = time()
    cuda_normalize[blockspergrid, threadsperblock](image8, mean, std) 
    t16  = time()
    print(f"result cuda_normalize second {image8[1, 0, :]}") 
    del image8


    print("=========================================")
    print(f"for normalize time {(t2 - t1)*1000} ms\n"
        f"numpy normalize time {(t4 - t3)*1000} ms\n"
        f"jit numpy normalize first time {(t6 - t5)*1000} ms, second time {(t8 - t7) * 1000} ms\n"
        f"jit for normalize first time {(t10 - t9) * 1000} ms, second time {(t12 - t11) * 1000} ms\n"
        f"cuda normalize first time {(t14 - t13) * 1000} ms, second time {(t16 - t15) * 1000} ms\n"
    )


if __name__ == "__main__":
    main()
