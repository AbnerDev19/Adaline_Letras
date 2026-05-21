import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extracao import carregar_dados
from adaline import Adaline

# ── Parâmetros ─────────────────────────────────────────────────────────────
TAXA_APRENDIZADO = 0.001   # menor = mais estável com mais dados
EPOCAS           = 300     # mais épocas para compensar mais amostras
N_LETRAS         = 26
# ───────────────────────────────────────────────────────────────────────────

def gerar_rotulo_one_vs_all(rotulos, letra_alvo):
    """Bipolar: +1 se for a letra alvo, -1 caso contrário."""
    return np.where(rotulos == letra_alvo, 1.0, -1.0)


def treinar_rede(entradas, rotulos):
    n_entradas = entradas.shape[1]
    redes      = []

    print(f"\n{'='*50}")
    print(f"  Treinamento One-vs-All")
    print(f"  Neurônios      : {N_LETRAS}")
    print(f"  Taxa aprendiz. : {TAXA_APRENDIZADO}")
    print(f"  Épocas         : {EPOCAS}")
    print(f"  Amostras treino: {len(entradas)}")
    print(f"  Vetor entrada  : {n_entradas} features")
    print(f"{'='*50}\n")

    for i in range(N_LETRAS):
        letra = chr(65 + i)
        n_pos = int(np.sum(rotulos == i))
        print(f"[{i+1:02d}/26] Neurônio letra {letra}  ({n_pos} amostra(s) positiva(s))")

        rede = Adaline(n_entradas, TAXA_APRENDIZADO)
        rotulos_bin = gerar_rotulo_one_vs_all(rotulos, i)
        rede.treinar(entradas, rotulos_bin, EPOCAS)
        redes.append(rede)
        print()

    return redes


def salvar_modelo(redes, pasta="modelos"):
    os.makedirs(pasta, exist_ok=True)

    todos_pesos = np.array([r.pesos for r in redes])
    todos_bias  = np.array([r.bias  for r in redes])

    np.save(os.path.join(pasta, "pesos.npy"), todos_pesos)
    np.save(os.path.join(pasta, "bias.npy"),  todos_bias)

    print(f"Modelo salvo em '{pasta}/pesos.npy' e '{pasta}/bias.npy'")


def avaliar(redes, entradas, rotulos):
    acertos = 0
    erros_det = []

    for x, rotulo_real in zip(entradas, rotulos):
        saidas  = [r.prever(x) for r in redes]
        previsto = int(np.argmax(saidas))

        if previsto == rotulo_real:
            acertos += 1
        else:
            erros_det.append((chr(65+rotulo_real), chr(65+previsto)))

    acuracia = acertos / len(entradas) * 100
    print(f"\nAcurácia no treino: {acertos}/{len(entradas)} = {acuracia:.1f}%")

    if erros_det:
        print("Erros detectados:")
        for real, prev in erros_det:
            print(f"  Real={real}  Previsto={prev}")

    return acuracia


if __name__ == "__main__":
    # Muda para o diretório raiz do projeto (onde fica a pasta 'dados/')
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(raiz)

    print("Carregando imagens...")
    entradas, rotulos = carregar_dados()

    if len(entradas) == 0:
        print("Nenhuma imagem encontrada. Verifique a pasta 'dados/treino/'.")
        sys.exit(1)

    redes = treinar_rede(entradas, rotulos)
    salvar_modelo(redes)
    avaliar(redes, entradas, rotulos)
