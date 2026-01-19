import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read a file from disk
# When reading a colour image, opencv reads it in the BGR colour space
# img = cv2.imread("images/boy.jpg", cv2.IMREAD_COLOR)
# img = cv2.imread("images/boy.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("images/boy.jpg", cv2.IMREAD_UNCHANGED)  # WITH ALPHA CHANNEL IF IT EXISTS


# Show the image in an emergent window
# cv2.imshow("image", img)
# cv2.waitKey(0)

# Write the image to disk
# When saving, it is transformed from BGR to RGB.
cv2.imwrite("images/boy_saved.jpg", img)
cv2.imwrite("images/boy_saved.png", img)

## Operaciones con las imágenes

# Acceder y cambiar valores de los pixeles
roi = img[0:100, 0:100]
# cv2.imshow("roi", roi)
# cv2.waitKey(0)

copia = img.copy()
copia[0:100, 0:100] = 0
# cv2.imshow("img changed", copia)
# cv2.waitKey(0)

# Acceder por canales
img_blue = img[:,:,0]
# cv2.imshow("img blue", img_blue)
# cv2.waitKey(0)

# Si quisiera que la imagen se viera en azul, tendría que mantener
# los otros dos canales, pero ponerlos a 0
copia = img.copy()
copia[:,:,1] = 0
copia[:,:,2] = 0
# cv2.imshow("img colour blue", copia)
# cv2.waitKey(0)

# Convertir la imagen de BGR a otro sistema de color o grayscale
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("img grayscale", img_gray)
# cv2.waitKey(0)

# Convertir a RGB
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# cv2.imshow("img RGB", img_rgb)
# cv2.waitKey(0)

# Usar split para separar los canales
b,g,r = cv2.split(img)
# cv2.imshow("blue", b)
# cv2.waitKey(0)

# Usar merge para volver a juntarlos
image_merge = cv2.merge([b, g, r])
# cv2.imshow("merged", image_merge)
# cv2.waitKey(0)

# Histogramas: cuantas veces se repite cada valor de intensidad en una imagen
colors = ["b", "g", "r"]
plt.figure()
for i, c in enumerate(colors):
    # Hay que pasar la imagen como una lista, indicar el canal para hacer el histograma
    # también el rango de valores (256 porque es abierto)
    img_hist = cv2.calcHist([img],[i], None, [256], [0,256])
    plt.plot(img_hist, color=c)
plt.show()

# La ecualización de histograma se pueden usar para mejorar el contraste
img_equalized = np.copy(img)
for channel in range(img.shape[2]):
    img_equalized[:,:,channel] = cv2.equalizeHist(img[:,:,channel])
cv2.imshow("equalized", img_equalized)
cv2.waitKey(0)

colors = ["b", "g", "r"]
plt.figure()
for i, c in enumerate(colors):
    # Hay que pasar la imagen como una lista, indicar el canal para hacer el histograma
    # también el rango de valores (256 porque es abierto)
    img_hist = cv2.calcHist([img_equalized],[i], None, [256], [0,256])
    plt.plot(img_hist, color=c)
plt.show()


# Normalización de imagenes.
# Se usa para tener los valores de intensidad en rangos más adecuados
# por ejemplo para modelos de ML
# Se normaliza en escala de grises y se vuelve a pasar a color.
# Otra opción es normalizar canal por canal
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_norm = cv2.normalize(img_gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
img_norm = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
cv2.imshow("normalized", img_norm)
cv2.waitKey(0)

