import pickle
from deepface import DeepFace
import argparse
import os

def registrar_nuevo_usuario(nombre, ruta_img):
    if not os.path.exists(ruta_img):
        print("La imagen no existe")
        return

    archivo_pkl = "embeddings.pkl"
    base_datos_vectores = {}

    if os.path.exists(archivo_pkl):
        with open(archivo_pkl,'rb') as f:
            base_datos_vectores = pickle.load(f)

    try:
        resultado = DeepFace.represent(img_path=ruta_img, 
                                        model_name="VGG-Face", 
                                        detector_backend="retinaface")
        base_datos_vectores[nombre] = resultado[0]["embedding"]

        with open(archivo_pkl, 'wb') as f:
            pickle.dump(base_datos_vectores, f)

        print(f"Se ha registrado a {nombre}")

    except Exception as e:
        print(f"Error al registrar : {e}")

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nombre")
    parser.add_argument("ruta_img")
    args = parser.parse_args()
    registrar_nuevo_usuario(args.nombre, args.ruta_img)