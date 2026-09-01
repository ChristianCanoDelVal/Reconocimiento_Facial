import pickle
from deepface import DeepFace
import argparse
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

def acceso_usuario(imagen):
    '''
    Recibe la imagen y en caso de detectar que es un trabajador devuelve su 
    nombre, en caso contrario devuelve None
    '''
    archivo_pkl = "embeddings.pkl"
    if not os.path.exists(archivo_pkl):
        return None
    
    with open(archivo_pkl,'rb') as f:
        base_datos_vectores = pickle.load(f)

    try:
        resultado = DeepFace.represent(img_path=imagen, 
                                        model_name="VGG-Face", 
                                        enforce_detection=False,
                                        detector_backend="skip")
        vec = resultado[0]["embedding"]
        v1 = np.array(vec) / np.linalg.norm(vec)

        min_dist = float("inf")
        min_nombre = None

        for nombre, embedding in base_datos_vectores.items():
            v2 = np.array(embedding) / np.linalg.norm(embedding)
            dist = np.linalg.norm(v1 - v2)
            if dist < min_dist:
                min_dist = dist
                min_nombre = nombre

            print(f"Comparando con {nombre}: Distancia = {dist}")

        umbral = 1
        if min_dist > umbral:
            return None

        return min_nombre

    except Exception as e:
        print(f"Error al comprobar si el usuario puede acceder : {e}")
        return None

def registrar_nuevo_usuario(nombre, ruta_img):
    '''
    Toma el nombre y ruta completa de la imagen del nuevo usuario y lo registra en la base de datos
    '''
    if not os.path.exists(ruta_img):
        print("La imagen no existe")
        return

    archivo_pkl = "embeddings.pkl"
    base_datos_vectores = {}

    if os.path.exists(archivo_pkl):
        with open(archivo_pkl,'rb') as f:
            base_datos_vectores = pickle.load(f)

    try:
        base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
        options = vision.FaceDetectorOptions(base_options=base_options)

        with vision.FaceDetector.create_from_options(options) as detector:
            imagen = cv2.imread(ruta_img)
            rgb_img = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            resultado = detector.detect(mp_image)

            if not resultado.detections:
                print(f"Error al registrar")
                return
            
            deteccion = resultado.detections[0]
            confianza = deteccion.categories[0].score

            if confianza < 0.85:
                print(f"Error al registrar, no se detecta bien a la persona")
                return
            
            bbox = deteccion.bounding_box
            x = bbox.origin_x
            y = bbox.origin_y
            w = bbox.width
            h = bbox.height

            rostro_persona = imagen[y:y+h,x:x+w]
                
            if rostro_persona.size <= 0:
                print(f"Error al registrar, no se detecta bien a la persona")
                return
            
            resultado = DeepFace.represent(img_path=rostro_persona, 
                                            model_name="VGG-Face", 
                                            enforce_detection=False,
                                            detector_backend="skip")
            base_datos_vectores[nombre] = resultado[0]["embedding"]

            with open(archivo_pkl, 'wb') as f:
                pickle.dump(base_datos_vectores, f)

            print(f"Se ha registrado a {nombre}")

    except Exception as e:
        print(f"Error al registrar : {e}")
        return

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nombre")
    parser.add_argument("ruta_img")
    args = parser.parse_args()
    registrar_nuevo_usuario(args.nombre, args.ruta_img)