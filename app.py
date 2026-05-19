from flask import Flask, request, jsonify, render_template_string
import numpy as np
import cv2
import base64
import os

app = Flask(__name__)

pesos = np.load("modelos/pesos.npy")
bias  = np.load("modelos/bias.npy")

def saida_linear(w, b, x):
    return np.dot(w, x) + b

def prever(x):
    saidas = [saida_linear(pesos[i], bias[i], x) for i in range(26)]
    return int(np.argmax(saidas)), saidas

def preprocessar(img_array):
    gray        = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    _, binaria  = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    redim       = cv2.resize(binaria, (28, 28))
    normalizada = redim / 255.0
    return normalizada.flatten()

HTML = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Reconhecimento de Letras — Adaline</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', sans-serif;
      background: #0f0f1a;
      color: #e0e0e0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 30px 20px;
    }

    header {
      text-align: center;
      margin-bottom: 30px;
    }

    header h1 { font-size: 2rem; color: #7c6bff; letter-spacing: 1px; }
    header p  { color: #888; margin-top: 6px; font-size: 0.95rem; }

    .card {
      background: #1a1a2e;
      border: 1px solid #2a2a4a;
      border-radius: 16px;
      padding: 28px;
      width: 100%;
      max-width: 500px;
      margin-bottom: 20px;
    }

    .card h2 {
      font-size: 0.85rem;
      color: #aaa;
      margin-bottom: 16px;
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    /* Área de upload */
    .upload-area {
      border: 2px dashed #3a3a6a;
      border-radius: 12px;
      padding: 40px 20px;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.2s, background 0.2s;
    }

    .upload-area:hover, .upload-area.drag { 
      border-color: #7c6bff; 
      background: #1f1f3a;
    }

    .upload-area .icone { font-size: 3rem; margin-bottom: 10px; }
    .upload-area p { color: #888; font-size: 0.9rem; }
    .upload-area span { color: #7c6bff; font-weight: bold; }

    #file-input { display: none; }

    /* Preview da imagem */
    .preview-container {
      display: none;
      margin-top: 16px;
      text-align: center;
    }

    .preview-container img {
      max-width: 100%;
      max-height: 250px;
      border-radius: 10px;
      border: 2px solid #2a2a4a;
      object-fit: contain;
      background: white;
    }

    .nome-arquivo {
      margin-top: 8px;
      font-size: 0.8rem;
      color: #666;
    }

    .botoes { 
      display: flex;
      gap: 12px;
      margin-top: 16px;
    }

    button {
      flex: 1;
      padding: 12px;
      border: none;
      border-radius: 10px;
      font-size: 1rem;
      font-weight: bold;
      cursor: pointer;
      transition: opacity 0.2s;
    }

    button:hover { opacity: 0.85; }
    button:disabled { opacity: 0.4; cursor: not-allowed; }

    #btn-reconhecer { background: #7c6bff; color: white; }
    #btn-limpar     { background: #2a2a4a; color: #ccc; }

    /* Resultado */
    .resultado-box { text-align: center; }

    .letra-grande {
      font-size: 6rem;
      font-weight: bold;
      color: #7c6bff;
      line-height: 1;
      margin: 10px 0;
      min-height: 100px;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .confianca { font-size: 0.9rem; color: #888; margin-top: 8px; }

    .barra-container { margin-top: 20px; }
    .barra-container h2 {
      font-size: 0.85rem;
      color: #aaa;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 10px;
    }

    .barra-linha {
      display: flex;
      align-items: center;
      margin-bottom: 5px;
      gap: 8px;
      font-size: 0.8rem;
    }

    .barra-letra { width: 18px; color: #ccc; font-weight: bold; }

    .barra-fundo {
      flex: 1;
      height: 10px;
      background: #2a2a4a;
      border-radius: 5px;
      overflow: hidden;
    }

    .barra-fill {
      height: 100%;
      border-radius: 5px;
      background: #7c6bff;
      transition: width 0.4s ease;
    }

    .barra-fill.destaque { background: #00d4aa; }

    .status {
      text-align: center;
      color: #555;
      font-size: 0.9rem;
      margin-top: 10px;
    }

    footer {
      margin-top: 30px;
      color: #444;
      font-size: 0.8rem;
      text-align: center;
    }
  </style>
</head>
<body>

<header>
  <h1>🧠 Reconhecimento de Letras</h1>
  <p>Rede Neural Adaline — Introdução à Inteligência Artificial</p>
</header>

<div class="card">
  <h2>📁 Envie uma imagem da letra</h2>

  <div class="upload-area" id="upload-area" onclick="document.getElementById('file-input').click()">
    <div class="icone">🖼️</div>
    <p>Clique para selecionar ou <span>arraste uma imagem</span></p>
    <p style="margin-top:6px;font-size:0.75rem">PNG, JPG, BMP — fundo branco, letra escura</p>
  </div>

  <input type="file" id="file-input" accept="image/*" onchange="carregarImagem(event)">

  <div class="preview-container" id="preview-container">
    <img id="preview-img" src="" alt="preview">
    <div class="nome-arquivo" id="nome-arquivo"></div>
  </div>

  <div class="botoes">
    <button id="btn-reconhecer" onclick="reconhecer()" disabled>Reconhecer</button>
    <button id="btn-limpar" onclick="limpar()">Limpar</button>
  </div>

  <p class="status" id="status">Selecione uma imagem para começar</p>
</div>

<div class="card resultado-box">
  <h2>🎯 Resultado</h2>
  <div class="letra-grande" id="letra-resultado">—</div>
  <div class="confianca" id="confianca-texto"></div>

  <div class="barra-container">
    <h2>Pontuação por letra</h2>
    <div id="barras"></div>
  </div>
</div>

<footer>Trabalho 1 — Adaline | IFB</footer>

<script>
  let imagemBase64 = null;

  // Drag and drop
  const uploadArea = document.getElementById('upload-area');
  uploadArea.addEventListener('dragover',  e => { e.preventDefault(); uploadArea.classList.add('drag'); });
  uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag'));
  uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('drag');
    const file = e.dataTransfer.files[0];
    if (file) processarArquivo(file);
  });

  function carregarImagem(event) {
    const file = event.target.files[0];
    if (file) processarArquivo(file);
  }

  function processarArquivo(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
      imagemBase64 = e.target.result;

      document.getElementById('preview-img').src = imagemBase64;
      document.getElementById('preview-container').style.display = 'block';
      document.getElementById('nome-arquivo').textContent = file.name;
      document.getElementById('btn-reconhecer').disabled = false;
      document.getElementById('status').textContent = 'Imagem carregada! Clique em Reconhecer.';

      // Limpa resultado anterior
      document.getElementById('letra-resultado').textContent = '—';
      document.getElementById('confianca-texto').textContent = '';
      document.getElementById('barras').innerHTML = '';
    };
    reader.readAsDataURL(file);
  }

  function limpar() {
    imagemBase64 = null;
    document.getElementById('file-input').value = '';
    document.getElementById('preview-container').style.display = 'none';
    document.getElementById('preview-img').src = '';
    document.getElementById('nome-arquivo').textContent = '';
    document.getElementById('btn-reconhecer').disabled = true;
    document.getElementById('letra-resultado').textContent = '—';
    document.getElementById('confianca-texto').textContent = '';
    document.getElementById('barras').innerHTML = '';
    document.getElementById('status').textContent = 'Selecione uma imagem para começar';
  }

  async function reconhecer() {
    if (!imagemBase64) return;
    document.getElementById('status').textContent = 'Analisando...';
    document.getElementById('btn-reconhecer').disabled = true;

    const resp = await fetch('/prever', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imagem: imagemBase64 })
    });

    const data = await resp.json();
    document.getElementById('btn-reconhecer').disabled = false;

    if (data.erro) {
      document.getElementById('status').textContent = '⚠️ ' + data.erro;
      return;
    }

    document.getElementById('letra-resultado').textContent = data.letra;
    document.getElementById('confianca-texto').textContent = `Neurônio ${data.letra} teve a maior ativação`;
    document.getElementById('status').textContent = '✅ Reconhecido!';

    // Barras
    const barrasDiv = document.getElementById('barras');
    barrasDiv.innerHTML = '';
    const saidas = data.saidas;
    const maxVal = Math.max(...saidas);
    const minVal = Math.min(...saidas);
    const range  = maxVal - minVal || 1;

    saidas.forEach((s, i) => {
      const letra    = String.fromCharCode(65 + i);
      const pct      = ((s - minVal) / range) * 100;
      const destaque = i === data.indice ? 'destaque' : '';
      barrasDiv.innerHTML += `
        <div class="barra-linha">
          <span class="barra-letra">${letra}</span>
          <div class="barra-fundo">
            <div class="barra-fill ${destaque}" style="width:${pct.toFixed(1)}%"></div>
          </div>
          <span style="color:#666;width:50px;text-align:right">${s.toFixed(2)}</span>
        </div>`;
    });
  }
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/prever", methods=["POST"])
def prever_rota():
    try:
        data      = request.get_json()
        img_b64   = data["imagem"].split(",")[1]
        img_bytes = base64.b64decode(img_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img       = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        vetor          = preprocessar(img)
        indice, saidas = prever(vetor)
        letra          = chr(65 + indice)

        return jsonify({
            "letra":  letra,
            "indice": indice,
            "saidas": [round(float(s), 4) for s in saidas]
        })

    except Exception as e:
        return jsonify({"erro": str(e)})

if __name__ == "__main__":
    print("=" * 45)
    print("  Adaline — Reconhecimento de Letras")
    print("  Acesse: http://localhost:5000")
    print("=" * 45)
    app.run(debug=True)