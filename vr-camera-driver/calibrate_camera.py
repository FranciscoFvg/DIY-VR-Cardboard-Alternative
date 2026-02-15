import cv2
import numpy as np
import glob
import os

# Configuração
CHECKERBOARD = (9, 6)  # Número de cantos internos (largura, altura)
SQUARE_SIZE = 0.025    # Tamanho do quadrado em metros (2.5cm)

# Critérios de terminação para refinamento de cantos
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Preparar pontos 3D do tabuleiro
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Arrays para armazenar pontos
objpoints = []  # Pontos 3D no mundo real
imgpoints = []  # Pontos 2D na imagem

print("=== Calibração de Câmera ===")
print(f"Tabuleiro: {CHECKERBOARD[0]}x{CHECKERBOARD[1]}")
print(f"Tamanho do quadrado: {SQUARE_SIZE*100}cm")
print("\nPressione ESPAÇO para capturar imagem")
print("Pressione 'c' para calcular calibração")
print("Pressione ESC para sair\n")

# Abrir câmera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

images_captured = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    display = frame.copy()
    
    # Encontrar cantos do tabuleiro
    ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret_corners:
        # Refinar posição dos cantos
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Desenhar
        cv2.drawChessboardCorners(display, CHECKERBOARD, corners2, ret_corners)
        cv2.putText(display, "Tabuleiro detectado! Pressione ESPACO", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.putText(display, f"Imagens capturadas: {images_captured}", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Calibracao', display)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        break
    elif key == ord(' ') and ret_corners:  # ESPAÇO
        objpoints.append(objp)
        imgpoints.append(corners2)
        images_captured += 1
        print(f"Imagem {images_captured} capturada")
    elif key == ord('c') and images_captured >= 10:  # 'c' para calibrar
        print("\nCalculando calibração...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        
        if ret:
            print("\n=== Calibração Concluída ===")
            print("\nMatriz da câmera:")
            print(mtx)
            print("\nCoeficientes de distorção:")
            print(dist)
            
            # Calcular erro de reprojeção
            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
            
            mean_error /= len(objpoints)
            print(f"\nErro médio de reprojeção: {mean_error}")
            
            # Salvar em arquivo
            np.savez('camera_calibration.npz', 
                    mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
            print("\nSalvo em: camera_calibration.npz")
            
            # Gerar código C++ para copiar
            print("\n=== Copie para main.cpp ===")
            print("cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << ")
            print(f"    {mtx[0,0]:.2f}, {mtx[0,1]:.2f}, {mtx[0,2]:.2f},")
            print(f"    {mtx[1,0]:.2f}, {mtx[1,1]:.2f}, {mtx[1,2]:.2f},")
            print(f"    {mtx[2,0]:.2f}, {mtx[2,1]:.2f}, {mtx[2,2]:.2f});")
            print("\ncv::Mat distCoeffs = (cv::Mat_<double>(5,1) << ")
            print(f"    {dist[0,0]:.6f}, {dist[0,1]:.6f}, {dist[0,2]:.6f}, {dist[0,3]:.6f}, {dist[0,4]:.6f});")
            break

cap.release()
cv2.destroyAllWindows()

if images_captured < 10:
    print("\nNecessário pelo menos 10 imagens para calibração!")
