import cv2
import numpy as np
import os

TAMANHO = 28  # 28x28 pixels
PASTA_TREINO = "dados/treino/"

def processar_imagem(caminho):
    # 1. Lê a imagem
    img = cv2.imread(caminho)

    # 2. Converte para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Binariza (pixels viram 0 ou 255)
    _, binaria = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 4. Inverte: fundo = 0, letra = 1
    invertida = cv2.bitwise_not(binaria)

    # 5. Redimensiona para 28x28
    redimensionada = cv2.resize(invertida, (TAMANHO, TAMANHO))

    # 6. Normaliza para valores 0 e 1
    normalizada = redimensionada / 255.0

    # 7. Transforma em vetor (784 valores)
    vetor = normalizada.flatten()

    return vetor


def carregar_dados():
    entradas = []
    rotulos = []

    for i in range(1, 27):  # 1 a 26
        caminho = os.path.join(PASTA_TREINO, f"{i}.png")

        if not os.path.exists(caminho):
            print(f"Imagem não encontrada: {caminho}")
            continue

        vetor = processar_imagem(caminho)
        entradas.append(vetor)
        rotulos.append(i - 1)  # 0=A, 1=B, ..., 25=Z

        letra = chr(65 + i - 1)  # 65 = 'A' em ASCII
        print(f"Processada: {i}.png → Letra {letra}")

    entradas = np.array(entradas)
    rotulos = np.array(rotulos)

    print(f"\nTotal de imagens carregadas: {len(entradas)}")
    print(f"Tamanho de cada vetor: {entradas.shape[1]}")

    return entradas, rotulos


# Teste rápido
if __name__ == "__main__":
    entradas, rotulos = carregar_dados()
    print("\nPrimeiros 5 valores do vetor da letra A:")
    print(entradas[0][:5])