import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extracao import carregar_dados
from adaline import Adaline

# ── Parâmetros ─────────────────────────────────────────────────────────────
TAXA_APRENDIZADO = 0.00005  # Ponto doce para evitar a explosão matemática
EPOCAS           = 5000     
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
        
        # --- EQUILÍBRIO DE CLASSES (Oversampling) ---
        idx_positivos = np.where(rotulos_bin == 1.0)[0]
        idx_negativos = np.where(rotulos_bin == -1.0)[0]

        if len(idx_positivos) > 0 and len(idx_negativos) > 0:
            fator_multiplicacao = len(idx_negativos) // len(idx_positivos)
            idx_pos_balanceado = np.repeat(idx_positivos, fator_multiplicacao)
            
            indices_balanceados = np.concatenate([idx_pos_balanceado, idx_negativos])
            np.random.shuffle(indices_balanceados)

            entradas_treino = entradas[indices_balanceados]
            rotulos_treino  = rotulos_bin[indices_balanceados]
        else:
            entradas_treino = entradas
            rotulos_treino  = rotulos_bin

        rede.treinar(entradas_treino, rotulos_treino, EPOCAS)
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


def salvar_grafico(redes, pasta="modelos"):
    """Gera o gráfico da Curva de Aprendizado e salva em PNG"""
    print("Gerando gráfico da Curva de Aprendizado...")
    plt.figure(figsize=(12, 8))
    
    for i, rede in enumerate(redes):
        letra = chr(65 + i)
        plt.plot(rede.historico_erro, label=letra, alpha=0.6)
        
    plt.title(f"Curva de Aprendizado - Adaline ({EPOCAS} Épocas | LR: {TAXA_APRENDIZADO})")
    plt.xlabel("Épocas")
    plt.ylabel("Erro Médio (MSE)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    caminho_grafico = os.path.join(pasta, "curva_aprendizado.png")
    plt.savefig(caminho_grafico)
    plt.close()
    print(f"Gráfico salvo em: '{caminho_grafico}'")


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
    return acuracia


if __name__ == "__main__":
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(raiz)

    print("Carregando imagens...")
    entradas, rotulos = carregar_dados()

    if len(entradas) == 0:
        print("Nenhuma imagem encontrada. Verifique a pasta 'dados/treino/'.")
        sys.exit(1)

    redes = treinar_rede(entradas, rotulos)
    salvar_modelo(redes)
    salvar_grafico(redes)
    avaliar(redes, entradas, rotulos)