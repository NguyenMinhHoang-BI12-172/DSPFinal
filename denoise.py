import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load noisy images
image = cv2.imread('bear.jpg', 0)
noisy1 = cv2.imread('noisy_gaussian_image.jpg', 0)
noisy2 = cv2.imread('noisy_salt_pepper_image.jpg', 0)


def apply_gaussian_filtering(image):
    F = np.fft.fft2(image)
    Fshift = np.fft.fftshift(F)

    # Filter: Low pass filter
    M, N = image.shape
    H = np.zeros((M, N), dtype=np.float32)
    D0 = 50
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            if D <= D0:
                H[u, v] = 1
            else:
                H[u, v] = 0

    # Ideal Low Pass Filtering
    Gshift = Fshift * H

    # Inverse Fourier Transform
    G = np.fft.ifftshift(Gshift)
    filtered_image = np.abs(np.fft.ifft2(G))
    return filtered_image


def displayNoisyImages(row, col, image1, image2, image3):
    fig, axs = plt.subplots(row, col, figsize=(15, 5))
    fig.tight_layout()
    fig.suptitle("Original and Noisy Images comparison", fontsize=16, fontweight='bold')
    axs[0].imshow(image1, cmap='gray')
    axs[0].set_title('Original')
    axs[0].axis('off')
    axs[1].imshow(image2, cmap='gray')
    axs[1].set_title('Noisy image')
    axs[1].axis('off')
    axs[2].imshow(image3, cmap='gray')
    axs[2].set_title('Filtered image')
    axs[2].axis('off')
    plt.show()


img1 = apply_gaussian_filtering(noisy1)
img2 = apply_gaussian_filtering(noisy2)
displayNoisyImages(1, 3, image, noisy1, img1)
displayNoisyImages(1, 3, image, noisy2, img2)
