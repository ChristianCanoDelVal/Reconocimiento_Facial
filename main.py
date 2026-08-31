import cv2
from mtcnn import MTCNN
from mtcnn.utils.plotting import plot
import matplotlib.pyplot as plt

def procesar_video(ruta_video):
    '''
    Toma la ruta de un video o 0 en caso de ser la camara del propio pc
    y la muestra por pantalla
    '''
    cap = cv2.VideoCapture(ruta_video)
    mtcnn = MTCNN()

    #Comprobamos que se haya abierto correctamente
    if not cap.isOpened():
        print("Error al abrir el video")
        return -1

    #Creamos una ventana con ciertas medidas 
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video", 1280, 720)

    #Calculamos el delay para mostar cada frame para mantener la misma velocidad de video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 0:
        delay = int(1000/fps)
    else:
        delay = 30

    contador_frames = 0
    caras_detectadas = []
    #Mostramos cada frame en bucle hasta que se produzca un error o fin de video
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al visualizar el video.")
            break

        contador_frames += 1

        if contador_frames % 5 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            caras_detectadas = mtcnn.detect_faces(rgb_frame)

        for cara in caras_detectadas:
            confianza = cara["confidence"]

            if confianza > 0.9:  
                x, y, w, h = cara["box"]
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                texto = f"Confianza del {confianza*100:.2f}"
                cv2.putText(frame,texto, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        cv2.imshow("Video",frame)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    #Liberamos y destruimos todas las ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    ruta_video = "prueba.mp4"
    procesar_video(ruta_video)
