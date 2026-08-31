import cv2

def procesar_video(ruta_video):
    '''
    Toma la ruta de un video o 0 en caso de ser la camara del propio pc
    y la muestra por pantalla
    '''
    cap = cv2.VideoCapture(ruta_video)

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

    #Mostramos cada frame en bucle hasta que se produzca un error o fin de video
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al visualizar el video.")
            break
        cv2.imshow("Video",frame)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    #Liberamos y destruimos todas las ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    ruta_video = "prueba.mp4"
    procesar_video(ruta_video)
