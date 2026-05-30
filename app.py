import os
from flask import Flask, request, jsonify, render_template_string, send_from_directory
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Carregamento dos parâmetros do modelo Adaline
try:
    pesos = np.load("modelos/pesos.npy")
    bias  = np.load("modelos/bias.npy")
except FileNotFoundError:
    print("Aviso: Arquivos de configuração do modelo não foram localizados na pasta 'modelos'.")
    pesos, bias = None, None

def saida_linear(w, b, x):
    return np.dot(w, x) + b

def prever(x):
    saidas = [saida_linear(pesos[i], bias[i], x) for i in range(26)]
    return int(np.argmax(saidas)), saidas

def preprocessar(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binaria = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binaria)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        letra_recortada = binaria[y:y+h, x:x+w]
        tamanho_max = max(w, h)
        pad_y = (tamanho_max - h) // 2
        pad_x = (tamanho_max - w) // 2
        margem = int(tamanho_max * 0.15)
        letra_centralizada = cv2.copyMakeBorder(
            letra_recortada, pad_y + margem, pad_y + margem, 
            pad_x + margem, pad_x + margem, cv2.BORDER_CONSTANT, value=0
        )
    else:
        letra_centralizada = binaria

    redim = cv2.resize(letra_centralizada, (32, 32), interpolation=cv2.INTER_AREA)
    normalizada = (redim / 255.0) * 2.0 - 1.0
    return normalizada.flatten()

HTML = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Painel de Controle - Adaline Letras</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg-main: #ffffff;
      --text-primary: #37352f;
      --text-secondary: rgba(55, 53, 47, 0.65);
      --border-accent: rgba(55, 53, 47, 0.12);
      --bg-hover: rgba(55, 53, 47, 0.04);
      --graph-neutral: rgba(55, 53, 47, 0.2);
      --graph-active: #37352f;
    }

    body { 
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      background-color: var(--bg-main); 
      color: var(--text-primary);
      margin: 0;
      padding: 40px 24px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    .main-header {
      width: 100%;
      max-width: 1140px;
      margin-bottom: 40px;
      border-bottom: 1px solid var(--border-accent);
      padding-bottom: 16px;
    }

    .main-header h1 {
      font-size: 24px;
      font-weight: 600;
      margin: 0 0 8px 0;
    }

    .main-header p {
      font-size: 14px;
      color: var(--text-secondary);
      margin: 0;
    }

    .dashboard-grid {
      display: flex;
      gap: 32px;
      width: 100%;
      max-width: 1140px;
      align-items: flex-start;
    }

    @media (max-width: 900px) {
      .dashboard-grid {
        flex-direction: column;
      }
    }

    .panel-card {
      background: var(--bg-main);
      border: 1px solid var(--border-accent);
      border-radius: 4px;
      padding: 24px;
      box-shadow: rgba(15, 15, 15, 0.02) 0px 1px 3px;
      box-sizing: border-box;
    }

    .col-left {
      flex: 1;
    }

    .col-right {
      flex: 1.2;
    }

    .panel-card h2 {
      font-size: 16px;
      font-weight: 600;
      margin-top: 0;
      margin-bottom: 20px;
      border-bottom: 1px solid var(--border-accent);
      padding-bottom: 8px;
    }

    .metrics-summary {
      display: flex;
      gap: 16px;
      margin-bottom: 24px;
    }

    .metric-box {
      flex: 1;
      padding: 12px;
      border: 1px solid var(--border-accent);
      border-radius: 4px;
      background: var(--bg-hover);
    }

    .metric-label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-secondary);
      margin-bottom: 4px;
    }

    .metric-value {
      font-size: 20px;
      font-weight: 600;
    }

    .learning-curve-wrapper {
      width: 100%;
      border: 1px solid var(--border-accent);
      border-radius: 4px;
      overflow: hidden;
    }

    .learning-curve-img {
      width: 100%;
      height: auto;
      display: block;
    }

    .upload-zone {
      border: 1px dashed var(--border-accent);
      border-radius: 4px;
      padding: 40px 20px;
      text-align: center;
      cursor: pointer;
      font-size: 14px;
      color: var(--text-secondary);
      transition: background 0.2s;
    }

    .upload-zone:hover {
      background: var(--bg-hover);
    }

    #file-input { display: none; }

    .preview-section {
      display: none;
      justify-content: space-around;
      margin-top: 24px;
    }

    .preview-box {
      text-align: center;
    }

    .preview-box img {
      max-width: 110px;
      max-height: 110px;
      border: 1px solid var(--border-accent);
      border-radius: 4px;
      padding: 4px;
      background: #ffffff;
    }

    .preview-box .pixelated-view {
      image-rendering: pixelated;
      width: 110px;
      height: 110px;
      background: #000000;
    }

    .preview-box span {
      display: block;
      font-size: 12px;
      color: var(--text-secondary);
      margin-top: 6px;
    }

    .action-button {
      width: 100%;
      padding: 10px 16px;
      background: var(--text-primary);
      color: #ffffff;
      border: none;
      border-radius: 4px;
      font-size: 14px;
      font-weight: 500;
      margin-top: 24px;
      cursor: pointer;
      transition: opacity 0.2s;
    }

    .action-button:hover {
      opacity: 0.9;
    }

    .action-button:disabled {
      background: var(--border-accent);
      color: var(--text-secondary);
      cursor: not-allowed;
    }

    .results-display {
      display: none;
      margin-top: 28px;
      border-top: 1px solid var(--border-accent);
      padding-top: 24px;
    }

    .output-header {
      text-align: center;
      margin-bottom: 20px;
    }

    .output-label {
      font-size: 12px;
      text-transform: uppercase;
      color: var(--text-secondary);
    }

    .output-letter {
      font-size: 56px;
      font-weight: 700;
      margin: 4px 0;
    }

    .chart-frame {
      position: relative;
      height: 260px;
      width: 100%;
    }

    .error-fallback {
      font-size: 13px;
      color: var(--text-secondary);
      padding: 32px 16px;
      text-align: center;
    }
  </style>
</head>
<body>

<div class="main-header">
  <h1>Sistema de Reconhecimento Adaline</h1>
  <p>Análise matricial de caracteres baseada em Redes Neurais de Camada Única</p>
</div>

<div class="dashboard-grid">

  <div class="panel-card col-left">
    <h2>Métricas do Modelo Treinado</h2>
    <div class="metrics-summary">
      <div class="metric-box">
        <div class="metric-label">Arquitetura</div>
        <div class="metric-value">26 Neurônios</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Resolução</div>
        <div class="metric-value">32 × 32 px</div>
      </div>
    </div>
    <div class="learning-curve-wrapper">
      <img src="/modelos/curva_aprendizado.png" class="learning-curve-img" alt="Gráfico de Convergência" onerror="this.style.display='none'; document.getElementById('image-error').style.display='block';">
      <div id="image-error" class="error-fallback" style="display: none;">
        O arquivo 'curva_aprendizado.png' não foi localizado no diretório de modelos.
      </div>
    </div>
  </div>

  <div class="panel-card col-right">
    <h2>Classificação de Amostras</h2>
    <div class="upload-zone" onclick="document.getElementById('file-input').click()">
      Clique para selecionar ou arraste o arquivo de imagem
    </div>
    <input type="file" id="file-input" accept="image/*" onchange="gerenciarArquivo(event)">

    <div class="preview-section" id="preview-section">
      <div class="preview-box">
        <img id="view-original" src="">
        <span>Amostra Original</span>
      </div>
      <div class="preview-box">
        <img id="view-processed" class="pixelated-view" src="">
        <span>Vetor Normalizado</span>
      </div>
    </div>

    <button id="btn-submit" class="action-button" onclick="executarAnalise()" disabled>Processar Classificação</button>

    <div class="results-display" id="results-display">
      <div class="output-header">
        <div class="output-label">Caractere Identificado</div>
        <div class="output-letter" id="target-letter">-</div>
      </div>
      <div class="chart-frame">
        <canvas id="activationChart"></canvas>
      </div>
    </div>
  </div>

</div>

<script>
  let dadosBase64 = null;
  let chartObject = null;

  function gerenciarArquivo(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function(e) {
      dadosBase64 = e.target.result;
      document.getElementById('view-original').src = dadosBase64;
      document.getElementById('view-processed').src = "";
      document.getElementById('preview-section').style.display = 'flex';
      document.getElementById('btn-submit').disabled = false;
      document.getElementById('results-display').style.display = 'none';
    };
    reader.readAsDataURL(file);
  }

  async function executarAnalise() {
    const button = document.getElementById('btn-submit');
    button.disabled = true;
    button.innerText = "Processando matriz...";

    try {
      const response = await fetch('/prever', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imagem: dadosBase64 })
      });
      
      const result = await response.json();
      
      if (result.erro) {
        alert("Erro operacional: " + result.erro);
        return;
      }

      document.getElementById('target-letter').textContent = result.letra;
      document.getElementById('view-processed').src = result.imagem_vista;
      document.getElementById('results-display').style.display = 'block';

      configurarGrafico(result.saidas, result.indice);

    } catch (err) {
      console.error(err);
      alert("Erro na comunicação com o servidor local.");
    } finally {
      button.disabled = false;
      button.innerText = "Processar Classificação";
    }
  }

  function configurarGrafico(saidas, indiceVencedor) {
    const canvasElement = document.getElementById('activationChart').getContext('2d');
    
    // Normalização das saídas lineares para percentual relativo de acurácia por tentativa
    const maxVal = Math.max(...saidas);
    const minVal = Math.min(...saidas);
    const amplitude = maxVal - minVal || 1;
    const dadosPercentuais = saidas.map(val => ((val - minVal) / amplitude) * 100);
    
    const classesLabels = Array.from({length: 26}, (_, i) => String.fromCharCode(65 + i));
    const coresColunas = classesLabels.map((_, i) => i === indiceVencedor ? '#37352f' : 'rgba(55, 53, 47, 0.15)');

    if (chartObject) {
      chartObject.destroy();
    }

    chartObject = new Chart(canvasElement, {
      type: 'bar',
      data: {
        labels: classesLabels,
        datasets: [{
          data: dadosPercentuais,
          backgroundColor: coresColunas,
          borderRadius: 2,
          borderSkipped: false
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function(item) {
                return ` Ativação Linear: ${saidas[item.dataIndex].toFixed(4)}`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: 'rgba(55, 53, 47, 0.6)', font: { size: 11 } }
          },
          y: {
            grid: { color: 'rgba(55, 53, 47, 0.06)' },
            ticks: { display: false },
            border: { display: false }
          }
        }
      }
    });
  }
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

# Rota estática para fornecimento de arquivos de modelos (Gráficos persistidos)
@app.route("/modelos/<path:filename>")
def fornecer_modelo(filename):
    return send_from_directory("modelos", filename)

@app.route("/prever", methods=["POST"])
def prever_rota():
    if pesos is None or bias is None:
        return jsonify({"erro": "Parâmetros do modelo não carregados em memória."})
    try:
        data = request.get_json()
        img_b64 = data["imagem"].split(",")[1]
        img_bytes = base64.b64decode(img_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Processamento vetorial e predição linear
        vetor = preprocessar(img)
        indice, saidas = prever(vetor)
        letra = chr(65 + indice)

        # Reconstrução da amostragem 32x32 mapeada para 0-255
        vetor_2d = vetor.reshape((32, 32))
        img_reconstruida = ((vetor_2d + 1.0) / 2.0 * 255.0).astype(np.uint8)
        
        _, buffer = cv2.imencode('.png', img_reconstruida)
        img_vista_b64 = "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "letra": letra,
            "indice": int(indice),
            "saidas": [round(float(s), 4) for s in saidas],
            "imagem_vista": img_vista_b64
        })
    except Exception as e:
        return jsonify({"erro": str(e)})

if __name__ == "__main__":
    app.run(debug=True)