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

def compressFQ_image(img, P, Q, filtre, n=8):
    nrows, ncols, _ = img.shape
    compressed = np.zeros_like(img)
    for c in range(3):
        for i in range(0, nrows, n):
            for j in range(0, ncols, n):
                block = img[i:i+n, j:j+n, c]
                D = np.dot(P, np.dot(block, P.T))
                temp = np.zeros((n, n))
                for m in range(min(filtre, n)):
                    for l in range(min(filtre, n) - m):
                        temp[m, l] = D[m, l]
                compressed_block = np.round(temp / Q)
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
    decompressed = np.clip(decompressed + 128, 0, 255)
    return decompressed

def calculate_compression_rate(compressed, nrows, ncols):
    number = np.count_nonzero(compressed)
    return 100 - (round(number / (nrows * ncols * 3) * 100))

def display_images(original, decompressed_images, compression_rates, q_labels, filtre):
    plt.figure(figsize=(20, 10))
    
    # Image originale
    plt.subplot(2, 3, 1)
    plt.title("Image originale")
    plt.imshow(original + 128)

    # Images décompressées
    for idx, (decompressed, compression_rate) in enumerate(zip(decompressed_images, compression_rates)):
        plt.subplot(2, 3, idx + 2)
        plt.title(f"Compressée {q_labels[idx]} filtre : {filtre} ")
        plt.imshow(decompressed)
        # Ajout du taux de compression en dessous de l'image
        plt.text(0.5, -0.15, f"compression : {compression_rate}%", ha='center', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "image.jpg"
    img, nrows, ncols = initialisation_image(image_path)

    P = create_dct_matrix()
    filtre = 6

    Q_faible = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 13, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])

    Q_classique = Q_faible * 4

    Q_aggressif = Q_faible * 8

    Q_extreme = Q_faible * 15

    Q_max = Q_faible * 40

    list_q = [Q_faible, Q_classique, Q_aggressif, Q_extreme, Q_max]
    q_labels = ["Q classique", "Q * 4", "Q * 8", "Q * 15", "Q * 40"]

    decompressed_images = []
    compression_rates = []

    for Q in list_q:
        before = np.count_nonzero(img)
        compressed = compressFQ_image(img, P, Q, filtre)
        after = np.count_nonzero(compressed)
        difference_before_after = round((before-after)/(nrows*ncols))*100
        compression_rate = calculate_compression_rate(compressed, nrows, ncols)
        decompressed = decompress_image(compressed, P, Q)
        decompressed_images.append(decompressed)
        compression_rates.append(compression_rate)
        print("non zeros avant:" ,before , "\n" , "apres:" , after, "\n")

    display_images(img, decompressed_images, compression_rates, q_labels, filtre)