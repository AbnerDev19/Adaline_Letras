import cv2
import numpy as np
import os

TAMANHO = 28          # 28x28 pixels
PASTA_TREINO = "dados/treino/"

def preprocessar_imagem(caminho):
    """
    Pré-processamento IDÊNTICO ao do app.py para garantir consistência
    entre treino e inferência.
    """
    img = cv2.imread(caminho)
    if img is None:
        raise ValueError(f"Não foi possível ler a imagem: {caminho}")

    # 1. Escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Desfoque Gaussiano leve
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Binarização de Otsu + inversão (fundo preto, traço branco)
    _, binaria = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Encontrar bounding box da letra
    coords = cv2.findNonZero(binaria)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        letra_recortada = binaria[y:y+h, x:x+w]

        # Criar quadrado perfeito + margem de 15%
        tamanho_max = max(w, h)
        pad_y = (tamanho_max - h) // 2
        pad_x = (tamanho_max - w) // 2
        margem = int(tamanho_max * 0.15)

        letra_centralizada = cv2.copyMakeBorder(
            letra_recortada,
            pad_y + margem, pad_y + margem,
            pad_x + margem, pad_x + margem,
            cv2.BORDER_CONSTANT, value=0
        )
    else:
        letra_centralizada = binaria

    # 5. Redimensionar para 28x28
    redim = cv2.resize(letra_centralizada, (TAMANHO, TAMANHO), interpolation=cv2.INTER_AREA)

    # 6. Normalizar 0.0–1.0
    normalizada = redim / 255.0

    return normalizada.flatten()


def carregar_dados():
    """
    Carrega dados de pasta com subdiretórios por letra:
      dados/treino/A/img1.png, img2.png, ...
      dados/treino/B/img1.png, ...
      ...
    Suporta também o formato legado (N.png direto na pasta raiz).
    """
    entradas = []
    rotulos  = []

    for i in range(26):
        letra     = chr(65 + i)          # A, B, C, ...
        pasta_letra = os.path.join(PASTA_TREINO, letra)

        # ----- NOVO: pasta por letra -----
        if os.path.isdir(pasta_letra):
            arquivos = sorted([
                f for f in os.listdir(pasta_letra)
                if f.lower().endswith(('.png', '.jpg', '.bmp', '.jpeg'))
            ])

            if not arquivos:
                print(f"[AVISO] Pasta {pasta_letra} está vazia — letra {letra} ignorada.")
                continue

            for nome in arquivos:
                caminho = os.path.join(pasta_letra, nome)
                try:
                    vetor = preprocessar_imagem(caminho)
                    entradas.append(vetor)
                    rotulos.append(i)
                    print(f"  Processada: {letra}/{nome} → Letra {letra}")
                except Exception as e:
                    print(f"  [ERRO] {caminho}: {e}")

        # ----- LEGADO: N.png na raiz -----
        else:
            caminho = os.path.join(PASTA_TREINO, f"{i+1}.png")
            if os.path.exists(caminho):
                try:
                    vetor = preprocessar_imagem(caminho)
                    entradas.append(vetor)
                    rotulos.append(i)
                    print(f"  Processada: {i+1}.png → Letra {letra}")
                except Exception as e:
                    print(f"  [ERRO] {caminho}: {e}")
            else:
                print(f"[AVISO] Nenhuma imagem encontrada para a letra {letra}.")

    entradas = np.array(entradas)
    rotulos  = np.array(rotulos)

    print(f"\nTotal de amostras carregadas : {len(entradas)}")
    if len(entradas) > 0:
        print(f"Tamanho de cada vetor       : {entradas.shape[1]}")

    return entradas, rotulos


if __name__ == "__main__":
    entradas, rotulos = carregar_dados()
    print("\nPrimeiros 5 valores do vetor da letra A:")
    print(entradas[0][:5])
