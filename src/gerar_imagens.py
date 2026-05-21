from PIL import Image, ImageDraw, ImageFont
import os

PASTA_TREINO = "dados/treino/"
PASTA_TESTE  = "dados/teste/"
TAMANHO_IMG  = 100

os.makedirs(PASTA_TREINO, exist_ok=True)
os.makedirs(PASTA_TESTE,  exist_ok=True)

def desenhar_letra(letra, tamanho_fonte=70):
    img  = Image.new("RGB", (TAMANHO_IMG, TAMANHO_IMG), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        fonte = ImageFont.truetype("arial.ttf", tamanho_fonte)
    except:
        fonte = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), letra, font=fonte)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (TAMANHO_IMG - w) // 2
    y = (TAMANHO_IMG - h) // 2

    draw.text((x, y), letra, fill=(0, 0, 0), font=fonte)
    return img

print("Gerando imagens...\n")

for i, letra in enumerate(map(chr, range(65, 91))):
    idx = i + 1
    img = desenhar_letra(letra)
    img.save(os.path.join(PASTA_TREINO, f"{idx}.png"))
    print(f"  {letra} → {idx}.png")

print("\n✓ 26 imagens geradas em dados/treino/")