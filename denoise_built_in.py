import numpy as np
import random
import cv2
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('bear.jpg', 0)


# Defining Gaussian noise function
def add_gaussian_noise(image, mean, std, intensity):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noise *= intensity
    noisy_image = cv2.add(image, noise.astype(np.uint8))
    return noisy_image


def add_salt_pepper_noise(image, salt_ratio, pepper_ratio, block_size):
    if len(image.shape) == 2:  # Grayscale image
        h, w = image.shape
        c = 1
    else:  # Color image
        h, w, c = image.shape

    noisy_salt_pepper_image = np.copy(image)
    num_salt_blocks = int(salt_ratio * (h // block_size) * (w // block_size))
    num_pepper_blocks = int(pepper_ratio * (h // block_size) * (w // block_size))

    # Add salt noise
    for _ in range(num_salt_blocks):
        y = np.random.randint(0, h // block_size)
        x = np.random.randint(0, w // block_size)
        if c == 1:
            noisy_salt_pepper_image[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size] = 255
        else:
            noisy_salt_pepper_image[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size, :] = 255

    # Add pepper noise
    for _ in range(num_pepper_blocks):
        y = np.random.randint(0, h // block_size)
        x = np.random.randint(0, w // block_size)
        if c == 1:
            noisy_salt_pepper_image[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size] = 0
        else:
            noisy_salt_pepper_image[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size, :] = 0

    return noisy_salt_pepper_image


# Denoise function (using built in median blur filter)
def denoise_salt_pepper_image(img, k):
    de_img = cv2.medianBlur(img, k)
    return de_img


def denoise_gaussian_image(img, k, sigma):
    de_img = cv2.GaussianBlur(img, (k, k), sigma)
    de_img1 = cv2.medianBlur(de_img, 5)
    return de_img1


# Displaying noisy images function
def displayNoisyImages(row, col, image1, image2, image3):
    fig, axs = plt.subplots(row, col, figsize=(15, 5))
    fig.tight_layout()
    fig.suptitle("Original and Noisy Images comparison", fontsize=16, fontweight='bold')
    axs[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original')
    axs[1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Noisy image (Gaussian)')
    axs[2].imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))
    axs[2].set_title('Noisy image (Salt and Pepper)')
    plt.show()


# Displaying denoise images
def displayDenoise(row, col, image1, image2, image3):
    fig, axs = plt.subplots(row, col, figsize=(15, 5))
    fig.tight_layout()
    fig.suptitle("Original and Denoise Images comparison", fontsize=16, fontweight='bold')
    axs[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original')
    axs[1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Denoise image (Gaussian)')
    axs[2].imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))
    axs[2].set_title('Denoise image (Salt and Pepper)')
    plt.show()


# Printing accuracy percentage and loss of the filtered image (denoise) compare to the orginal
def percentage(image1, image2):
    different = cv2.absdiff(image1, image2)
    if len(different.shape) > 2 and different.shape[2] == 3:
        # Convert the difference image to grayscale if it is color (BGR format)
        gray = cv2.cvtColor(different, cv2.COLOR_BGR2GRAY)
    else:
        gray = different

    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pixels = threshold.shape[0] * threshold.shape[1]
    changed_pixels = cv2.countNonZero(threshold)
    accuracy_percent = ((pixels - changed_pixels) / pixels) * 100
    loss_percent = (changed_pixels / pixels) * 100
    print("Accuracy in percentage is:" + str(accuracy_percent) + "%")
    print("Total loss in percentage is:" + str(loss_percent) + "%")


# Adding the noise
noisy_gaussian_image = add_gaussian_noise(image, 0, 50, intensity=0.02)
noisy_salt_pepper_image = add_salt_pepper_noise(image, 0.01, 0.01, 3)

# Storing images with noise added
cv2.imwrite('noisy_gaussian_image.jpg', noisy_gaussian_image)
cv2.imwrite('noisy_salt_pepper_image.jpg', noisy_salt_pepper_image)

# Denoise
gaussian = denoise_gaussian_image(noisy_gaussian_image, 25, 1.0)
salt_pepper = denoise_salt_pepper_image(noisy_salt_pepper_image, 5)

# Printing the percentages
percentage(image, gaussian)
percentage(image, salt_pepper)

# Displaying images
displayNoisyImages(1, 3, image, noisy_gaussian_image, noisy_salt_pepper_image)
displayDenoise(1, 3, image, gaussian, salt_pepper)
