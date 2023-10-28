import cv2

# Verifique se o OpenCV foi compilado com suporte a GPU
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    # Crie um objeto de captura de vídeo com suporte a GPU
    cap = cv2.cuda.VideoCapture(0)

    while True:
        # Leia um quadro do dispositivo de captura de vídeo com suporte a GPU
        frame_gpu = cv2.cuda_GpuMat()
        if cap.read(frame_gpu):
            # Converta o quadro de GPU para a matriz numpy para processamento
            frame_cpu = frame_gpu.download()

            # Realize o processamento de imagem na matriz numpy
            # ...

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    print("A aceleração por GPU não está disponível.")
