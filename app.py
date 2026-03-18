from flask import Flask, render_template
import os
from datetime import datetime

app = Flask(__name__)

@app.route("/")
def dashboard():
    imagens = []
    pasta_fotos = "fotos"

    for arquivo in os.listdir(pasta_fotos):
        if arquivo.endswith(".jpg"):
            caminho = f"{pasta_fotos}/{arquivo}"
            try:
                timestamp = int(arquivo.replace("foto_alerta_", "").replace(".jpg", ""))
                datahora = datetime.fromtimestamp(timestamp)
                data = datahora.strftime("%d/%m/%Y")
                hora = datahora.strftime("%H:%M:%S")
            except:
                data, hora = "Desconhecida", "Desconhecida"

            imagens.append({
                "caminho": caminho,
                "data": data,
                "hora": hora
            })

    imagens.sort(key=lambda x: x["data"] + x["hora"], reverse=True)
    return render_template("index.html", imagens=imagens)

if __name__ == "__main__":
    app.run(debug=True)