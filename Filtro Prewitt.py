import cv2
import numpy as np

def filtro_prewitt(imagem):
    # Converter a imagem para escala de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Aplicar o filtro Prewitt nas direções x e y
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

    gradiente_x = cv2.filter2D(imagem_cinza, -1, kernel_x)
    gradiente_y = cv2.filter2D(imagem_cinza, -1, kernel_y)

    # Calcular a magnitude do gradiente
    magnitude = np.sqrt(np.square(gradiente_x) + np.square(gradiente_y))

    # Converter a magnitude para o tipo de dados np.uint8
    magnitude = magnitude.astype(np.uint8)

    # Normalizar a magnitude para o intervalo [0, 255]
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    return magnitude

# Carregar a imagem
imagem = cv2.imread('C:/Users/User/Downloads/Original.jpg')

# Aplicar o filtro Prewitt
imagem_filtrada = filtro_prewitt(imagem)

# Exibir a imagem original e a imagem filtrada
cv2.imshow('Imagem Original', imagem)
cv2.imshow('Imagem Filtrada (Prewitt)', imagem_filtrada)
cv2.waitKey(0)
cv2.destroyAllWindows()