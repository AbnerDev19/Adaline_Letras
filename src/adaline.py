import numpy as np


class Adaline:
    def __init__(self, n_entradas, taxa_aprendizado=0.01):
        # Inicialização de He: pesos pequenos para estabilidade
        self.pesos = np.random.uniform(-0.1, 0.1, n_entradas)
        self.bias  = 0.0
        self.taxa_aprendizado = taxa_aprendizado
        self.historico_erro   = []

    def saida_linear(self, x):
        return np.dot(self.pesos, x) + self.bias

    def funcao_degrau(self, y):
        return 1 if y >= 0 else -1

    def prever(self, x):
        return self.saida_linear(x)   # retorna valor contínuo para argmax no treino

    def prever_classe(self, x):
        return self.funcao_degrau(self.saida_linear(x))

    def treinar(self, entradas, rotulos_desejados, epocas=100):
        """
        Regra delta (Widrow-Hoff) com embaralhamento por época
        para melhor convergência.
        """
        n = len(entradas)
        indices = np.arange(n)

        for epoca in range(epocas):
            np.random.shuffle(indices)   # embaralha a ordem a cada época
            erro_total = 0.0

            for idx in indices:
                x = entradas[idx]
                t = rotulos_desejados[idx]

                y     = self.saida_linear(x)
                erro  = t - y

                self.pesos += self.taxa_aprendizado * erro * x
                self.bias  += self.taxa_aprendizado * erro

                erro_total += erro ** 2

            erro_medio = erro_total / n
            self.historico_erro.append(erro_medio)

            if (epoca + 1) % 20 == 0:
                print(f"    Época {epoca+1:>3}/{epocas} — Erro médio: {erro_medio:.6f}")
