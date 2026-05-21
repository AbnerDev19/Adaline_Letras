"""
avaliacao.py — Avalia o modelo treinado em um conjunto de imagens.

Uso:
    python src/avaliacao.py                  # avalia com dados de treino
    python src/avaliacao.py caminho/img.png  # testa uma imagem específica
"""
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from extracao import carregar_dados, preprocessar_imagem


def carregar_modelo(pasta="modelos"):
    pesos = np.load(os.path.join(pasta, "pesos.npy"))
    bias  = np.load(os.path.join(pasta, "bias.npy"))
    return pesos, bias


def prever_letra(pesos, bias, vetor):
    saidas = [np.dot(pesos[i], vetor) + bias[i] for i in range(26)]
    indice = int(np.argmax(saidas))
    return chr(65 + indice), indice, saidas


def avaliar_conjunto(pesos, bias, entradas, rotulos):
    print("\n=== Avaliação no conjunto de treino ===")
    acertos = 0
    for i, (x, r) in enumerate(zip(entradas, rotulos)):
        letra_prev, _, _ = prever_letra(pesos, bias, x)
        letra_real = chr(65 + r)
        ok = "✓" if letra_prev == letra_real else "✗"
        if letra_prev == letra_real:
            acertos += 1
        else:
            print(f"  Amostra {i+1:>3}: Real={letra_real}  Previsto={letra_prev}  {ok}")

    acuracia = acertos / len(entradas) * 100
    print(f"\nAcurácia: {acertos}/{len(entradas)} = {acuracia:.1f}%")
    return acuracia


def testar_imagem(pesos, bias, caminho):
    print(f"\n=== Teste: {caminho} ===")
    vetor = preprocessar_imagem(caminho)
    letra, indice, saidas = prever_letra(pesos, bias, vetor)

    print(f"Letra reconhecida: {letra}")
    print("\nTop 5 ativações:")
    ranking = sorted(enumerate(saidas), key=lambda x: x[1], reverse=True)[:5]
    for idx, val in ranking:
        l = chr(65 + idx)
        marker = " ◄" if idx == indice else ""
        print(f"  {l}: {val:+.4f}{marker}")


if __name__ == "__main__":
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(raiz)

    pesos, bias = carregar_modelo()

    if len(sys.argv) > 1:
        # Testa uma imagem específica
        caminho = sys.argv[1]
        if os.path.exists(caminho):
            testar_imagem(pesos, bias, caminho)
        else:
            print(f"Arquivo não encontrado: {caminho}")
    else:
        # Avalia conjunto de treino
        entradas, rotulos = carregar_dados()
        avaliar_conjunto(pesos, bias, entradas, rotulos)
