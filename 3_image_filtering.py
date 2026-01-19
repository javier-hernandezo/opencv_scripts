import cv2
import numpy as np
import matplotlib.pyplot as plt

# Leemos la imagen de muestra en grayscale (para tener solo un canal
img = cv2.imread("images/boy.jpg", cv2.IMREAD_GRAYSCALE)

# Convolución

# Definimos un kernel
kernel_size = 3
kernel = np.ones((kernel_size, kernel_size))

# Vamos posición por posición de la imagen, multiplicando y sumando
# Hay que tener en cuenta el stride, el padding y las dimensiones de la imagen final
stride = 1
padding = int((kernel_size - 1) / 2)


def convolution(img, stride, padding):
    img_output = np.copy(img)
    shape = img.shape
    img_padded = np.zeros((shape[0] + (2 * padding), shape[1] + (2 * padding)), dtype=np.uint8)
    img_padded[padding:-padding, padding:-padding] = img

    kernel_sum = np.sum(kernel)
    for i in range(padding, shape[0], stride):
        for j in range(padding, shape[1], stride):
            img_output[i, j] = np.sum(img_padded[i - padding:i + padding + 1, j - padding:j + padding + 1] * kernel) * (1/kernel_sum)
    return img_output


# img_output = convolution(img, stride, padding)
# cv2.imshow("img", img)
# cv2.imshow("convolution", img_output)
# cv2.waitKey(0)


# Filtros para suavizar la imagen
img = cv2.resize(img, (img.shape[1]//2,img.shape[0]//2))
# Blur
dst_blur = cv2.blur(img, (kernel_size, kernel_size))

# Gaussian
dst_gauss = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# Mediana
dst_median = cv2.medianBlur(img, kernel_size)

# Bilateral
dst_bilateral = cv2.bilateralFilter(img, kernel_size, int(kernel_size*2), int(kernel_size/2))

# Visualize the four images at once
# Primera fila (2 imágenes lado a lado)
fila1 = np.hstack((dst_blur, dst_gauss))
# Segunda fila
fila2 = np.hstack((dst_median, dst_bilateral))
# Cuadrícula 2x2
grid = np.vstack((fila1, fila2))

# cv2.imshow("Filtros suavizado", grid)
# cv2.waitKey(0)


# Filtros para deteccion de bordes


# Sobel
scale = 1
delta = 0
ddepth = cv2.CV_16S
kernel_size = 3

# Primero gaussian blur para reducir ruido
src = cv2.GaussianBlur(img, (3, 3), 0)

# La imagen tiene que estar en grayscale
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray = src

# Calcular el gradiente en x e y
# Gradient-X
grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=kernel_size, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

# Gradient-Y
# grad_y = cv.Scharr(gray,ddepth,0,1)
grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=kernel_size, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

# Combinar ambos gradientes con sus valores absolutos
# converting back to uint8
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

cv2.imshow("Sobel", grad)
cv2.waitKey(0)


ddepth = cv2.CV_16S
kernel_size = 5
# Apply Laplace function
dst = cv2.Laplacian(gray, ddepth, ksize=kernel_size)


# converting back to uint8
abs_dst = cv2.convertScaleAbs(dst)

 
cv2.imshow("Laplace", abs_dst)
cv2.waitKey(0)


# Canny Edge Detector
lowThreshold = 50
lowThresholdRatio = 2
kernel_size = 5
 
def CannyThreshold(gray, lowThreshold, lowThresholdRatio, kernel_size):
    img_blur = cv2.blur(gray, (3,3))
    detected_edges = cv2.Canny(img_blur, lowThreshold, lowThresholdRatio * lowThreshold, kernel_size)
    mask = detected_edges != 0
    dst = gray * (mask[:,:].astype(src.dtype))
    cv2.imshow("Canny", dst)
 
 
CannyThreshold(gray,lowThreshold, lowThresholdRatio, kernel_size)
cv2.waitKey()