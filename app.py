from flask import Flask, request, jsonify, render_template_string
import numpy as np
import cv2
import base64

app = Flask(__name__)

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
  <title>Reconhecimento de Letras</title>
  <style>
    body { font-family: sans-serif; background: #f7f7f5; display: flex; flex-direction: column; align-items: center; padding: 40px; }
    .card { background: white; padding: 32px; border-radius: 6px; width: 100%; max-width: 540px; margin-bottom: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; cursor: pointer; }
    #file-input { display: none; }
    
    .imagens-wrapper { display: none; justify-content: space-around; margin-top: 20px; text-align: center; }
    .img-box img { max-width: 150px; max-height: 150px; object-fit: contain; border: 1px solid #eee; padding: 5px;}
    .img-box .adaline-vision { image-rendering: pixelated; width: 128px; height: 128px; background: black; }
    .img-box p { font-size: 13px; color: #666; margin-bottom: 5px; }

    button { width: 100%; padding: 12px; margin-top: 10px; cursor: pointer; background: #333; color: white; border: none; border-radius: 4px;}
    button:disabled { background: #999; }
    .barra-linha { display: flex; align-items: center; margin-bottom: 8px; font-size: 14px; }
    .barra-letra { width: 25px; font-weight: bold; }
    .barra-fundo { flex: 1; height: 10px; background: #eee; border-radius: 5px; overflow: hidden; margin-right: 10px;}
    .barra-fill { height: 100%; background: #999; }
    .barra-fill.destaque { background: #333; }
  </style>
</head>
<body>

<div class="card">
  <h3>Envie uma Imagem</h3>
  <div class="upload-area" onclick="document.getElementById('file-input').click()">Clique para enviar uma imagem</div>
  <input type="file" id="file-input" accept="image/*" onchange="carregarImagem(event)">

  <div class="imagens-wrapper" id="imagens-wrapper">
    <div class="img-box">
      <p>Original</p>
      <img id="preview-img" src="">
    </div>
    <div class="img-box">
      <p>Visão da Rede (32x32)</p>
      <img id="preview-adaline" class="adaline-vision" src="">
    </div>
  </div>

  <button id="btn-reconhecer" onclick="reconhecer()" disabled>Analisar Imagem</button>
</div>

<div class="card" id="resultado-box" style="display:none; text-align: center;">
  <h2>Resultado: <span id="letra-resultado" style="font-size: 3em;"></span></h2>
  <div id="barras" style="text-align: left; margin-top: 20px;"></div>
</div>

<script>
  let imagemBase64 = null;

  function carregarImagem(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function(e) {
      imagemBase64 = e.target.result;
      document.getElementById('preview-img').src = imagemBase64;
      document.getElementById('preview-adaline').src = ""; // limpa anterior
      document.getElementById('imagens-wrapper').style.display = 'flex';
      document.getElementById('btn-reconhecer').disabled = false;
      document.getElementById('resultado-box').style.display = 'none';
    };
    reader.readAsDataURL(file);
  }

  async function reconhecer() {
    document.getElementById('btn-reconhecer').disabled = true;
    const resp = await fetch('/prever', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ imagem: imagemBase64 })
    });
    
    const data = await resp.json();
    document.getElementById('btn-reconhecer').disabled = false;
    
    document.getElementById('letra-resultado').textContent = data.letra;
    
    // Mostra a visão do Adaline
    document.getElementById('preview-adaline').src = data.imagem_vista;

    const barrasDiv = document.getElementById('barras');
    barrasDiv.innerHTML = '';
    const saidas = data.saidas;
    const maxVal = Math.max(...saidas);
    const minVal = Math.min(...saidas);
    const range  = maxVal - minVal || 1;

    saidas.forEach((s, i) => {
      const letra = String.fromCharCode(65 + i);
      const pct = ((s - minVal) / range) * 100;
      const destaque = i === data.indice ? 'destaque' : '';
      barrasDiv.innerHTML += `
        <div class="barra-linha">
          <span class="barra-letra">${letra}</span>
          <div class="barra-fundo"><div class="barra-fill ${destaque}" style="width:${pct.toFixed(1)}%"></div></div>
          <span>${s.toFixed(2)}</span>
        </div>`;
    });
    
    document.getElementById('resultado-box').style.display = 'block';
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
        return jsonify({"erro": "Modelo não carregado."})
    try:
        data = request.get_json()
        img_b64 = data["imagem"].split(",")[1]
        img_bytes = base64.b64decode(img_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Prever
        vetor = preprocessar(img)
        indice, saidas = prever(vetor)
        letra = chr(65 + indice)

        # -----------------------------------------------------------------
        # NOVO: Reconstruir a imagem 32x32 para enviar ao Frontend
        # O vetor está de -1.0 a 1.0. Vamos convertê-lo para 0 a 255.
        # -----------------------------------------------------------------
        vetor_2d = vetor.reshape((32, 32))
        img_reconstruida = ((vetor_2d + 1.0) / 2.0 * 255.0).astype(np.uint8)
        
        # Codificar em Base64 para exibir no HTML
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