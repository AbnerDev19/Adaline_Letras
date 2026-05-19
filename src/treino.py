import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extracao import carregar_dados
from adaline import Adaline

# Parâmetros
TAXA_APRENDIZADO = 0.01
EPOCAS = 100
N_LETRAS = 26

def gerar_rotulo_one_vs_all(rotulos, letra_alvo):
    """
    Para cada neurônio, gera saída bipolar:
    +1 se for a letra alvo, -1 caso contrário
    """
    return np.where(rotulos == letra_alvo, 1, -1)


def treinar_rede(entradas, rotulos):
    n_entradas = entradas.shape[1]  # 784
    redes = []

    print(f"\nIniciando treinamento One-vs-All")
    print(f"  Neurônios: {N_LETRAS}")
    print(f"  Taxa de aprendizado: {TAXA_APRENDIZADO}")
    print(f"  Épocas: {EPOCAS}")
    print(f"  Tamanho do vetor de entrada: {n_entradas}\n")

    for i in range(N_LETRAS):
        letra = chr(65 + i)
        print(f"--- Treinando neurônio {i+1}/26: Letra {letra} ---")

        # Cria um Adaline para essa letra
        rede = Adaline(n_entradas, TAXA_APRENDIZADO)

        # Gera rótulos: +1 para essa letra, -1 para as outras
        rotulos_binarios = gerar_rotulo_one_vs_all(rotulos, i)

        # Treina
        rede.treinar(entradas, rotulos_binarios, EPOCAS)

        redes.append(rede)

    return redes


def salvar_modelo(redes):
    os.makedirs("modelos", exist_ok=True)

    todos_pesos = np.array([r.pesos for r in redes])
    todos_bias = np.array([r.bias for r in redes])

    np.save("modelos/pesos.npy", todos_pesos)
    np.save("modelos/bias.npy", todos_bias)

    print("\nModelo salvo em modelos/pesos.npy e modelos/bias.npy")


if __name__ == "__main__":
    # 1. Carrega e pré-processa as imagens
    print("Carregando imagens...")
    entradas, rotulos = carregar_dados()

    # 2. Treina a rede
    redes = treinar_rede(entradas, rotulos)

    # 3. Salva os pesos
    salvar_modelo(redes)

    # 4. Teste rápido no próprio conjunto de treino
    print("\n--- Teste rápido no conjunto de treino ---")
    acertos = 0
    for i, (x, rotulo_real) in enumerate(zip(entradas, rotulos)):
        saidas = [r.prever(x) for r in redes]
        previsto = np.argmax(saidas)  # neurônio com maior saída linear

        letra_real = chr(65 + rotulo_real)
        letra_prevista = chr(65 + previsto)

        if previsto == rotulo_real:
            acertos += 1
            print(f"  {i+1}.png → Real: {letra_real} | Previsto: {letra_prevista} ✓")
        else:
            print(f"  {i+1}.png → Real: {letra_real} | Previsto: {letra_prevista} ✗")

    acuracia = acertos / len(entradas) * 100
    print(f"\nAcurácia no treino: {acertos}/{len(entradas)} = {acuracia:.1f}%")