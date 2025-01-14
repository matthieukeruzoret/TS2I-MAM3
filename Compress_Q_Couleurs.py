import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg

def initialisation_image(image_path):
    img = mpimg.imread(image_path)
    if len(img.shape) == 2 or img.shape[2] < 3:
        raise ValueError("L'image doit contenir trois canaux de couleur (RGB).")
    if img.dtype != np.uint8:
        img = (img*255)
    else :
        img = (img/np.max(img))*255
    img = np.array(img, dtype=int)
    nrows, ncols, _ = img.shape
    nrows -= nrows % 8
    ncols -= ncols % 8
    img = img[:nrows, :ncols, :]
    img_shift = img
    img_shift = img_shift -128
    return img_shift, nrows, ncols, img

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
    compressed = np.zeros_like(img)
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
    decompressed = np.zeros_like(compressed)
    for c in range(3):
        for i in range(0, nrows, n):
            for j in range(0, ncols, n):
                block = compressed[i:i+n, j:j+n, c]
                D_tilde = block * Q
                M_tilde = np.dot(P.T, np.dot(D_tilde, P))
                decompressed[i:i+n, j:j+n, c] = M_tilde
    decompressed = decompressed.astype(np.int32)
    decompressed = np.clip(decompressed + 128, 0, 255)
    return decompressed

def calcul_compression_pourcentage(compressed, nrows, ncols):
    number = np.count_nonzero(compressed)
    return 100 - (round(number / (nrows * ncols * 3) * 100))

def affichage_images(original, compressed, decompressed):
    plt.figure(figsize=(10, 8))

    # Affichage de l'image originale
    plt.subplot(2, 2, 1)
    plt.title("Image originale tronquée")
    plt.imshow(np.clip(original + 128, 0, 255))

    # Affichage des coefficients DCT compressés (normalisés pour la visualisation)
    plt.subplot(2, 2, 2)
    plt.title("Coefficients compressés")
    plt.imshow(np.clip(compressed + 128, 0, 255))


    # Affichage de l'image décompressée
    plt.subplot(2, 2, 3)
    plt.title("Image décompressée")
    plt.imshow(decompressed)

    plt.tight_layout()
    plt.show()

def calcul_erreur(M, Mdecomp, couleur):
    if couleur:
        err = np.linalg.norm(M - Mdecomp) / np.linalg.norm(M) * 100
    else:
        err = np.linalg.norm(M[:, :, 0] - Mdecomp[:, :, 0]) / np.linalg.norm(M[:, :, 0]) * 100
    print(f"Pourcentage d'erreur : {err:.2f}%")

def main():
    image_path = "Images/CrazyFrog.png"
    img_shift, nrows, ncols, img = initialisation_image(image_path)
    image_start = img.copy()

    P = create_dct_matrix()

    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 13, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])

    compressed = compress_image(img_shift, P, Q)
    compression_rate = calcul_compression_pourcentage(compressed, nrows, ncols)
    print(f"Taux de compression Q : {compression_rate}%")

    decompressed = decompress_image(compressed, P, Q)
    calcul_erreur(image_start,decompressed,True)
    affichage_images(img_shift, compressed, decompressed)

if __name__ == "__main__":
    main()
