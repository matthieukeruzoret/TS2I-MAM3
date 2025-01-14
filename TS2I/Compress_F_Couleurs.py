import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg

def initialize_image(image_path):
    img = mpimg.imread(image_path)
    if len(img.shape) == 2 or img.shape[2] < 3:
        raise ValueError("L'image doit contenir trois canaux de couleur (RGB).")
    img = np.array(img, dtype=int)
    nrows, ncols, _ = img.shape
    nrows -= nrows % 8
    ncols -= ncols % 8
    img = img[:nrows, :ncols, :]
    img = img - 128
    return img, nrows, ncols

def create_dct_matrix(n=8):
    P = np.zeros((n, n))
    for k in range(n):
        for i in range(n):
            if k == 0:
                P[k, i] = 1 / math.sqrt(n)
            else:
                P[k, i] = math.sqrt(2 / n) * math.cos((2 * i + 1) * k * math.pi / (2 * n))
    return P

def apply_filter(D, filtre, n=8):
    for m in range(n):
        for l in range(n):
            if m + l >= filtre:
                D[m, l] = 0
    return D

def compress_image(img, P, filtre, n=8):
    nrows, ncols, _ = img.shape
    compressed = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        for i in range(0, nrows, n):
            for j in range(0, ncols, n):
                block = img[i:i+n, j:j+n, c]
                D = np.dot(P, np.dot(block, P.T))
                D = apply_filter(D, filtre, n)
                compressed[i:i+n, j:j+n, c] = D
    return compressed

def decompress_image(compressed, P, n=8):
    nrows, ncols, _ = compressed.shape
    decompressed = np.zeros_like(compressed, dtype=np.float32)
    for c in range(3):
        for i in range(0, nrows, n):
            for j in range(0, ncols, n):
                block = compressed[i:i+n, j:j+n, c]
                M_tilde = np.dot(P.T, np.dot(block, P))
                decompressed[i:i+n, j:j+n, c] = M_tilde
    decompressed = np.clip(decompressed + 128, 0, 255).astype(np.uint8)
    return decompressed

def calculate_compression_rate(compressed, nrows, ncols):
    number = np.count_nonzero(compressed)
    return 100 - (round(number / (nrows * ncols * 3) * 100))

def calculate_frobenius_norm(image_start, image_end):
    norme = np.linalg.norm((image_start.astype(np.float32) - image_end.astype(np.float32)).ravel(), ord=2)
    return norme / np.linalg.norm(image_start.astype(np.float32))

def display_images(original, compressed, decompressed):
    plt.subplot(2, 2, 1)
    plt.title("Image originale tronquée")
    plt.imshow(original + 128)

    plt.subplot(2, 2, 2)
    plt.title("Image compressée")
    plt.imshow(np.clip(compressed + 128, 0, 255).astype(np.uint8))

    plt.subplot(2, 2, 3)
    plt.title("Image décompressée")
    plt.imshow(decompressed)

    plt.tight_layout()
    plt.show()

def main():
    image_path = "Images/Cat.jpg"
    img, nrows, ncols = initialize_image(image_path)
    image_start = img.copy()

    P = create_dct_matrix()

    filtre = 10
    compressed = compress_image(img, P, filtre)
    compression_rate = calculate_compression_rate(compressed, nrows, ncols)
    print(f"Taux de compression F : {compression_rate}%")

    decompressed = decompress_image(compressed, P)
    frobenius_norm = calculate_frobenius_norm(image_start, decompressed)
    print(f"Norme de Frobenius : {frobenius_norm}%")

    display_images(img, compressed, decompressed)

if __name__ == "__main__":
    main()
