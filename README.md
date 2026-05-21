# Adaline — Reconhecimento de Letras

Rede neural Adaline para reconhecimento das 26 letras do alfabeto (A–Z).

## Estrutura do Projeto

```
Adaline_Letras-main/
├── app.py                  # Servidor Flask (interface web)
├── src/
│   ├── adaline.py          # Implementação da rede Adaline
│   ├── extracao.py         # Carregamento e pré-processamento de imagens
│   ├── treino.py           # Script de treinamento
│   └── avaliacao.py        # Avaliação e teste do modelo
├── dados/
│   └── treino/
│       ├── A/              # Imagens de treino da letra A
│       │   ├── orig_1.png
│       │   └── orig_2.png
│       ├── B/              # Imagens de treino da letra B
│       │   └── ...
│       └── ...             # Uma pasta por letra (A–Z)
└── modelos/
    ├── pesos.npy           # Pesos treinados
    └── bias.npy            # Bias treinados
```

## Como Adicionar Novas Imagens

Para melhorar o reconhecimento, adicione mais imagens na pasta da letra correspondente:

```
dados/treino/A/minha_letra_a.png
dados/treino/B/minha_letra_b.png
```

**Requisitos das imagens:**
- Formatos aceitos: PNG, JPG, BMP
- Fundo claro (branco/cinza), letra escura — ou fundo escuro, letra clara
- A letra deve ocupar boa parte da imagem

Após adicionar as imagens, **retreine o modelo**:
```bash
python src/treino.py
```

## Como Executar

### 1. Instalar dependências
```bash
pip install flask numpy opencv-python
```

### 2. Treinar o modelo (se necessário)
```bash
python src/treino.py
```

### 3. Iniciar o servidor
```bash
python app.py
```
Acesse: http://localhost:5000

## Como Testar uma Imagem Específica
```bash
python src/avaliacao.py caminho/para/imagem.png
```

## Tecnologia
- **Adaline** (Adaptive Linear Neuron) com regra delta de Widrow-Hoff
- Estratégia **One-vs-All**: 26 neurônios, um por letra
- Pré-processamento: escala de cinza → desfoque Gaussiano → binarização Otsu → bounding box → 28×28 pixels → normalização
