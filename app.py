from flask import Flask, request, jsonify, render_template_string
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Carregamento do modelo
try:
    pesos = np.load("modelos/pesos.npy")
    bias  = np.load("modelos/bias.npy")
except FileNotFoundError:
    print("Atenção: Arquivos de pesos não encontrados. Certifique-se de que a pasta 'modelos' existe.")
    pesos, bias = None, None

def saida_linear(w, b, x):
    return np.dot(w, x) + b

def prever(x):
    saidas = [saida_linear(pesos[i], bias[i], x) for i in range(26)]
    return int(np.argmax(saidas)), saidas

def preprocessar(img_array):
    # 1. Converter para tons de cinza
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # 2. Desfoque Gaussiano leve para reduzir ruídos pontuais antes da binarização
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Binarização de Otsu (limiar automático) + Inversão (fundo preto, traço branco)
    _, binaria = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. Encontrar as coordenadas dos pixels não-nulos (rastrear a letra)
    coords = cv2.findNonZero(binaria)
    
    if coords is not None:
        # Extrair a caixa delimitadora (bounding box) exata da letra
        x, y, w, h = cv2.boundingRect(coords)
        letra_recortada = binaria[y:y+h, x:x+w]
        
        # Calcular o preenchimento (padding) para criar um quadrado perfeito
        tamanho_max = max(w, h)
        pad_y = (tamanho_max - h) // 2
        pad_x = (tamanho_max - w) // 2
        
        # Adicionar uma margem de 15% para que os traços não toquem as bordas da matriz
        margem = int(tamanho_max * 0.15)
        
        letra_centralizada = cv2.copyMakeBorder(
            letra_recortada, 
            pad_y + margem, pad_y + margem, 
            pad_x + margem, pad_x + margem, 
            cv2.BORDER_CONSTANT, value=0
        )
    else:
        # Caso a imagem seja completamente vazia/branca
        letra_centralizada = binaria

    # 5. Redimensionar para 28x28 usando interpolação de área (ideal para redução)
    redim = cv2.resize(letra_centralizada, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 6. Normalizar os valores dos pixels para a escala de 0.0 a 1.0 (formato esperado pela rede)
    normalizada = redim / 255.0
    return normalizada.flatten()

HTML = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Reconhecimento de Letras — Adaline</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background: #f7f7f5;
      color: #37352f;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 40px 20px;
    }

    header {
      text-align: center;
      margin-bottom: 40px;
      max-width: 600px;
    }

    header h1 { 
      font-size: 2.2rem; 
      color: #37352f; 
      font-weight: 700;
      letter-spacing: -0.02em;
      margin-bottom: 8px;
    }
    
    header p { 
      color: #787774; 
      font-size: 1rem; 
      font-weight: 400;
    }

    .card {
      background: #ffffff;
      border: 1px solid rgba(55, 53, 47, 0.09);
      border-radius: 6px;
      box-shadow: rgba(15, 15, 15, 0.05) 0px 0px 0px 1px, rgba(15, 15, 15, 0.1) 0px 2px 4px;
      padding: 32px;
      width: 100%;
      max-width: 540px;
      margin-bottom: 24px;
    }

    .card h2 {
      font-size: 0.9rem;
      color: #787774;
      margin-bottom: 20px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      font-weight: 600;
    }

    /* Área de upload */
    .upload-area {
      border: 1px dashed rgba(55, 53, 47, 0.2);
      border-radius: 6px;
      background: #fbfbfa;
      padding: 40px 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .upload-area:hover, .upload-area.drag { 
      border-color: #2eaadc;
      background: #f2f8fc;
    }

    .upload-area svg {
      width: 32px;
      height: 32px;
      fill: #9a9a97;
      margin-bottom: 12px;
    }

    .upload-area p { color: #37352f; font-size: 0.95rem; font-weight: 500; }
    .upload-area span { color: #2eaadc; }
    .upload-area .subtext { color: #9a9a97; font-size: 0.8rem; margin-top: 8px; font-weight: 400; }

    #file-input { display: none; }

    /* Preview da imagem */
    .preview-container {
      display: none;
      margin-top: 20px;
      text-align: center;
      background: #f7f7f5;
      padding: 16px;
      border-radius: 6px;
      border: 1px solid rgba(55, 53, 47, 0.09);
    }

    .preview-container img {
      max-width: 100%;
      max-height: 200px;
      object-fit: contain;
    }

    .nome-arquivo {
      margin-top: 12px;
      font-size: 0.85rem;
      color: #787774;
    }

    .botoes { 
      display: flex;
      gap: 12px;
      margin-top: 24px;
    }

    button {
      flex: 1;
      padding: 10px 16px;
      border: 1px solid transparent;
      border-radius: 4px;
      font-size: 0.95rem;
      font-weight: 500;
      font-family: inherit;
      cursor: pointer;
      transition: opacity 0.2s, transform 0.1s;
    }

    button:active { transform: scale(0.98); }
    button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

    #btn-reconhecer { 
      background: linear-gradient(135deg, #3a3b5c 0%, #1a1a2e 100%); 
      color: white; 
      box-shadow: rgba(15, 15, 15, 0.1) 0px 1px 2px;
    }
    
    #btn-limpar { 
      background: #ffffff; 
      color: #37352f; 
      border-color: rgba(55, 53, 47, 0.16);
    }
    
    #btn-limpar:hover { background: #f7f7f5; }

    /* Resultado */
    .resultado-box { text-align: center; }

    .letra-grande {
      font-size: 5rem;
      font-weight: 700;
      color: #37352f;
      line-height: 1;
      margin: 16px 0;
      min-height: 80px;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .confianca { font-size: 0.9rem; color: #787774; margin-top: 8px; }

    .barra-container { margin-top: 32px; text-align: left; }
    .barra-container h2 { margin-bottom: 16px; }

    .barra-linha {
      display: flex;
      align-items: center;
      margin-bottom: 8px;
      gap: 12px;
      font-size: 0.85rem;
    }

    .barra-letra { width: 20px; color: #37352f; font-weight: 600; text-align: center; }

    .barra-fundo {
      flex: 1;
      height: 8px;
      background: rgba(55, 53, 47, 0.09);
      border-radius: 4px;
      overflow: hidden;
    }

    .barra-fill {
      height: 100%;
      border-radius: 4px;
      background: rgba(55, 53, 47, 0.4);
      transition: width 0.4s ease;
    }

    .barra-fill.destaque { 
      background: linear-gradient(90deg, #3a3b5c 0%, #1a1a2e 100%); 
    }

    .valor-saida { color: #787774; width: 45px; text-align: right; font-variant-numeric: tabular-nums; }

    .status {
      text-align: center;
      color: #787774;
      font-size: 0.9rem;
      margin-top: 16px;
      min-height: 20px;
    }

    footer {
      margin-top: auto;
      color: #9a9a97;
      font-size: 0.85rem;
      text-align: center;
      padding-top: 20px;
    }
  </style>
</head>
<body>

<header>
  <h1>Reconhecimento de Letras</h1>
  <p>Rede Neural Adaline - Inteligência Artificial</p>
</header>

<div class="card">
  <h2>Envio de Imagem</h2>

  <div class="upload-area" id="upload-area" onclick="document.getElementById('file-input').click()">
    <svg viewBox="0 0 24 24">
      <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.36 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/>
    </svg>
    <p>Clique para selecionar ou <span>arraste uma imagem</span></p>
    <p class="subtext">Formatos suportados: PNG, JPG, BMP</p>
  </div>

  <input type="file" id="file-input" accept="image/*" onchange="carregarImagem(event)">

  <div class="preview-container" id="preview-container">
    <img id="preview-img" src="" alt="Visualização da imagem">
    <div class="nome-arquivo" id="nome-arquivo"></div>
  </div>

  <div class="botoes">
    <button id="btn-reconhecer" onclick="reconhecer()" disabled>Analisar Imagem</button>
    <button id="btn-limpar" onclick="limpar()">Limpar Dados</button>
  </div>

  <p class="status" id="status">Aguardando seleção de imagem.</p>
</div>

<div class="card resultado-box">
  <h2>Resultado da Análise</h2>
  <div class="letra-grande" id="letra-resultado">—</div>
  <div class="confianca" id="confianca-texto">Nenhuma análise realizada</div>

  <div class="barra-container">
    <h2>Níveis de Ativação</h2>
    <div id="barras"></div>
  </div>
</div>

<footer>Instituto Federal de Brasília — Trabalho Prático: Adaline</footer>

<script>
  let imagemBase64 = null;

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
      document.getElementById('status').textContent = 'Imagem carregada com sucesso. Pronto para análise.';

      document.getElementById('letra-resultado').textContent = '—';
      document.getElementById('confianca-texto').textContent = 'Nenhuma análise realizada';
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
    document.getElementById('confianca-texto').textContent = 'Nenhuma análise realizada';
    document.getElementById('barras').innerHTML = '';
    document.getElementById('status').textContent = 'Aguardando seleção de imagem.';
  }

  async function reconhecer() {
    if (!imagemBase64) return;
    document.getElementById('status').textContent = 'Processando análise...';
    document.getElementById('btn-reconhecer').disabled = true;

    try {
      const resp = await fetch('/prever', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imagem: imagemBase64 })
      });

      const data = await resp.json();
      document.getElementById('btn-reconhecer').disabled = false;

      if (data.erro) {
        document.getElementById('status').textContent = 'Erro de processamento: ' + data.erro;
        return;
      }

      document.getElementById('letra-resultado').textContent = data.letra;
      document.getElementById('confianca-texto').textContent = `Neurônio referente à letra ${data.letra} obteve a maior ativação.`;
      document.getElementById('status').textContent = 'Análise concluída.';

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
            <span class="valor-saida">${s.toFixed(2)}</span>
          </div>`;
      });
    } catch (e) {
      document.getElementById('status').textContent = 'Erro de comunicação com o servidor.';
      document.getElementById('btn-reconhecer').disabled = false;
    }
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
    if pesos is None or bias is None:
        return jsonify({"erro": "O modelo não foi carregado corretamente no servidor."})

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
            "indice": int(indice), # Convertido explicitamente para int padrão Python
            "saidas": [round(float(s), 4) for s in saidas]
        })

    except Exception as e:
        return jsonify({"erro": f"Erro interno: {str(e)}"})

if __name__ == "__main__":
    print("-" * 50)
    print("  Adaline - Reconhecimento de Letras")
    print("  Servidor ativo em: http://localhost:5000")
    print("-" * 50)
    app.run(debug=True)