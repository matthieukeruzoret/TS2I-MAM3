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

def apply_filter(D, filtre, n=8):
    for m in range(n):
        for l in range(n):
            if np.any(m + l >= filtre):  # Utilisation de np.any() ou np.all()
                D[m, l] = 0
    return D

def compress_image(img, P, filtre, n=8):
    nrows, ncols, _ = img.shape
    compressed = np.zeros_like(img)

    for c in range(3):  # Pour chaque canal de couleur
        for i in range(0, nrows, n):
            for j in range(0, ncols, n):
                block = img[i:i+n, j:j+n, c]  # matrice M
                D = np.dot(P, np.dot(block, P.T))
                temp = apply_filter(D, filtre)
                compressed[i:i+n, j:j+n, c] = temp

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
    decompressed = decompressed.astype(np.int32)
    decompressed = np.clip(decompressed + 128, 0, 255)
    return decompressed

def calcul_erreur(M, Mdecomp, couleur):
    if couleur:
        err = np.linalg.norm(M - Mdecomp) / np.linalg.norm(M) * 100
    else:
        err = np.linalg.norm(M[:, :, 0] - Mdecomp[:, :, 0]) / np.linalg.norm(M[:, :, 0]) * 100
    print(f"Pourcentage d'erreur : {err:.2f}%")

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

def main():
    image_path = "Images/Cat.jpg"
    img_shift, nrows, ncols, img = initialisation_image(image_path)
    image_start = img.copy()

    P = create_dct_matrix()

    compressed = compress_image(img_shift, P, 16)
    compression_rate = calcul_compression_pourcentage(compressed, nrows, ncols)
    print(f"Taux de compression F : {compression_rate}%")

    decompressed = decompress_image(compressed, P)
    calcul_erreur(image_start, decompressed, True)
    affichage_images(img_shift, compressed, decompressed)

if __name__ == "__main__":
    main()
