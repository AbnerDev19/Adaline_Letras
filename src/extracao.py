import cv2
import numpy as np
import os

TAMANHO = 32          # Alterado de 28 para 32 pixels
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

    # 5. Redimensionar para 32x32 usando a variável TAMANHO
    redim = cv2.resize(letra_centralizada, (TAMANHO, TAMANHO), interpolation=cv2.INTER_AREA)

    # 6. Normalizar de -1.0 a 1.0 (Bipolar - ideal para Adaline)
    normalizada = (redim / 255.0) * 2.0 - 1.0

    return normalizada.flatten()


def carregar_dados():
    """
    Carrega dados de pasta com subdiretórios por letra ou ficheiros na raiz.
    Suporta:
      - dados/treino/A/img1.png
      - dados/treino/1.png (A), 2.png (B), ...
      - dados/treino/a (1).png, b (1).png, ...
    """
    entradas = []
    rotulos  = []

    if not os.path.exists(PASTA_TREINO):
        print(f"[ERRO] A pasta {PASTA_TREINO} não existe!")
        return np.array(entradas), np.array(rotulos)

    # Lista todos os itens dentro da pasta de treino de uma vez só
    itens_raiz = os.listdir(PASTA_TREINO)

    for i in range(26):
        letra = chr(65 + i)              # 'A', 'B', 'C', ...
        letra_minuscula = letra.lower()  # 'a', 'b', 'c', ...
        pasta_letra = os.path.join(PASTA_TREINO, letra)

        # ----- 1. FORMATO: Pasta por letra (ex: dados/treino/A/) -----
        if os.path.isdir(pasta_letra):
            arquivos = sorted([
                f for f in os.listdir(pasta_letra)
                if f.lower().endswith(('.png', '.jpg', '.bmp', '.jpeg'))
            ])
            for nome in arquivos:
                caminho = os.path.join(pasta_letra, nome)
                try:
                    vetor = preprocessar_imagem(caminho)
                    entradas.append(vetor)
                    rotulos.append(i)
                    print(f"  Processada: {letra}/{nome} → Letra {letra}")
                except Exception as e:
                    print(f"  [ERRO] {caminho}: {e}")

        # ----- 2. FORMATO: Ficheiros soltos na raiz de dados/treino/ -----
        # Filtra apenas os ficheiros de imagem válidos na raiz
        arquivos_raiz = [
            f for f in itens_raiz
            if f.lower().endswith(('.png', '.jpg', '.bmp', '.jpeg')) and os.path.isfile(os.path.join(PASTA_TREINO, f))
        ]

        for nome in arquivos_raiz:
            caminho = os.path.join(PASTA_TREINO, nome)
            corresponde = False

            # Caso legado: 1.png -> A, 2.png -> B, etc.
            if nome.lower() == f"{i+1}.png":
                corresponde = True
            
            # Caso novo: começa com a letra correspondente (ex: "a (1).png" ou "d (3).png")
            elif nome.lower().startswith(letra_minuscula):
                nome_sem_ext = os.path.splitext(nome)[0].strip()
                # Garante que o padrão é a letra seguida de espaço, parêntese ou hífen 
                if len(nome_sem_ext) == 1 or nome_sem_ext[1] in [' ', '(', '_', '-']:
                    corresponde = True

            if corresponde:
                try:
                    vetor = preprocessar_imagem(caminho)
                    entradas.append(vetor)
                    rotulos.append(i)
                    print(f"  Processada na raiz: {nome} → Letra {letra}")
                except Exception as e:
                    print(f"  [ERRO] {caminho}: {e}")

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