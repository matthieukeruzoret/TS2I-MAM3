import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg

def initialisation_image(image_path):
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

def compress_image(img, P, Q, n=8):
    nrows, ncols, _ = img.shape
    compressed = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        for i in range(0, nrows, n):
            for j in range(0, ncols, n):
                block = img[i:i+n, j:j+n, c]
                D = np.dot(P, np.dot(block, P.T))
                compressed_block = np.round(D / Q)
                compressed[i:i+n, j:j+n, c] = compressed_block
    return compressed

def decompress_image(compressed, P, Q, n=8):
    nrows, ncols, _ = compressed.shape
    decompressed = np.zeros_like(compressed, dtype=np.float32)
    for c in range(3):
        for i in range(0, nrows, n):
            for j in range(0, ncols, n):
                block = compressed[i:i+n, j:j+n, c]
                D_tilde = block * Q
                M_tilde = np.dot(P.T, np.dot(D_tilde, P))
                decompressed[i:i+n, j:j+n, c] = M_tilde
    decompressed = np.clip(decompressed + 128, 0, 255).astype(np.uint8)
    return decompressed

def calculate_compression_rate(compressed, nrows, ncols):
    number = np.count_nonzero(compressed)
    return 100 - (round(number / (nrows * ncols * 3) * 100))

def display_images(original, compressed, decompressed):
    plt.subplot(2, 2, 1)
    plt.title("Image originale tronquée")
    plt.imshow(original + 128)

    plt.subplot(2, 2, 2)
    plt.title("Image compressée")
    plt.imshow(np.clip(compressed, 0, 255).astype(np.int8))

    plt.subplot(2, 2, 3)
    plt.title("Image décompressée")
    plt.imshow(decompressed)

    plt.tight_layout()
    plt.show()

def main():
    image_path = "Images/Cat.jpg"
    img, nrows, ncols = initialisation_image(image_path)

    P = create_dct_matrix()

    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 13, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])

    compressed = compress_image(img, P, Q)
    compression_rate = calculate_compression_rate(compressed, nrows, ncols)
    print(f"Taux de compression Q : {compression_rate}%")

    decompressed = decompress_image(compressed, P, Q)
    display_images(img, compressed, decompressed)

if __name__ == "__main__":
    main()
