import numpy as np


class Adaline:
    def __init__(self, n_entradas, taxa_aprendizado=0.01):
        # Pesos aleatórios pequenos + bias
        self.pesos = np.random.uniform(-0.5, 0.5, n_entradas)
        self.bias = np.random.uniform(-0.5, 0.5)
        self.taxa_aprendizado = taxa_aprendizado
        self.historico_erro = []

    def saida_linear(self, x):
        # y = w1*x1 + w2*x2 + ... + wn*xn + bias
        return np.dot(self.pesos, x) + self.bias

    def funcao_degrau(self, y):
        # Saída bipolar: +1 ou -1
        if y >= 0:
            return 1
        else:
            return -1

    def prever(self, x):
        y = self.saida_linear(x)
        return self.funcao_degrau(y)

    def treinar(self, entradas, rotulos_desejados, epocas=100):
        print(f"Treinando por {epocas} épocas...")

        for epoca in range(epocas):
            erro_total = 0

            for x, t in zip(entradas, rotulos_desejados):
                # 1. Calcula saída linear
                y = self.saida_linear(x)

                # 2. Calcula erro
                erro = t - y

                # 3. Atualiza pesos pela regra delta
                self.pesos += self.taxa_aprendizado * erro * x
                self.bias += self.taxa_aprendizado * erro

                # 4. Acumula erro quadrático
                erro_total += erro ** 2

            # Erro médio da época
            erro_medio = erro_total / len(entradas)
            self.historico_erro.append(erro_medio)

            if (epoca + 1) % 10 == 0:
                print(f"  Época {epoca + 1}/{epocas} — Erro médio: {erro_medio:.4f}")

        print("Treinamento concluído!")