import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from deepface import DeepFace

def procesar_video(ruta_video):
    '''
    Toma la ruta de un video o 0 en caso de ser la camara del propio pc
    y la muestra por pantalla
    '''
    cap = cv2.VideoCapture(ruta_video)
    base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)

    #Comprobamos que se haya abierto correctamente
    if not cap.isOpened():
        print("Error al abrir el video")
        return -1

    #Creamos una ventana con ciertas medidas 
    cv2.namedWindow("Control de acceso", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Control de acceso", 1280, 720)

    #Calculamos el delay para mostar cada frame para mantener la misma velocidad de video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 0:
        delay = int(1000/fps)
    else:
        delay = 30

    with vision.FaceDetector.create_from_options(options) as detector:
        #Mostramos cada frame en bucle hasta que se produzca un error o fin de video
        while True:
            exito, frame = cap.read()
            if not exito:
                print("Error al visualizar el video.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            resultado = detector.detect(mp_image)

            if resultado.detections:
                for deteccion in resultado.detections:
                    confianza = deteccion.categories[0].score
                    if confianza > 0.9:
                        bbox = deteccion.bounding_box
                        x = bbox.origin_x
                        y = bbox.origin_y
                        w = bbox.width
                        h = bbox.height
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
                        texto = f"Confianza del {confianza*100:.2f}%"
                        cv2.putText(frame, texto, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            cv2.imshow("Control de acceso",frame)

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

    #Liberamos y destruimos todas las ventanas
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ruta_video = "prueba.mp4"
    procesar_video(ruta_video)
