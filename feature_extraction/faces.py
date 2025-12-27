# -*- coding: utf-8 -*-
import datetime
import cv2
import numpy as np
import math

from facenet_pytorch import MTCNN
from skimage import color, io

import matplotlib.pyplot as plt

# função retorna faces de uma imagem
def get_faces_mtcnn(path, device):

    pimg = io.imread(path)  
    if len(pimg.shape) == 2: 
        pimg = color.gray2rgb(pimg)
    
    dy = int(pimg.shape[0] * 0.05)
    dx = int(pimg.shape[1] * 0.05)
    pimg = cv2.copyMakeBorder(pimg, dy, dy, dx, dx,
                              cv2.BORDER_CONSTANT, value=[0, 0, 0])

    
    mtcnn = MTCNN(keep_all=True, device=device)
    boxes, conf, marks = mtcnn.detect(pimg, landmarks=True)
    
    faces = []
    if boxes is None:
        return faces
    
    result = [{'box': boxes[i], 'confidence': conf[i], 
               'keypoints': marks[i]}  for i in range(len(boxes))]
    
    LEFT_EYE, RIGHT_EYE = 0, 1
    
    for r in result:
#         if r['confidence'] <= 0.8: continue
            
        bounding_box = r['box']
        kp = r['keypoints'].astype(int)
        
        # coordenadas da detecção na imagem original x1, y1, x2, y2
        x0 = max(bounding_box[0], 0)
        y0 = max(bounding_box[1], 0)
        x1 = min(bounding_box[2], pimg.shape[1])
        y1 = min(bounding_box[3], pimg.shape[0])
        
        w, h = x1-x0, y1-y0

        # aumento da área detectada
        y0n = max(y0 - int(h * 0.6), 0)
        y1n = min(y1 + int(h * 0.6), pimg.shape[0])
        x0n = max(x0 - int(w * 0.6), 0)
        x1n = min(x1 + int(w * 0.6), pimg.shape[1])

        coord = (x0-dx, y0-dy, x1-dx, y1-dy)

        y0n, y1n, x0n, x1n = int(y0n), int(y1n), int(x0n), int(x1n) 
        img_face = pimg[y0n:y1n, x0n:x1n]

        # calcula centro da imagem detectada original - sem aumento
        cy = int((y0 + y1) / 2) - y0n
        cx = int((x0 + x1) / 2) - x0n

        # rotaciona imagem
        at = math.atan2(kp[RIGHT_EYE][1] - kp[LEFT_EYE][1],
                        kp[RIGHT_EYE][0] - kp[LEFT_EYE][0])
        gr = math.degrees(at)
        # print(gr)

        # cria uma borda
        bordv = int(0.15 * (y1n - y0n) / 2)
        bordh = int(0.15 * (x1n - x0n) / 2)

        imgr = cv2.copyMakeBorder(img_face, bordv, bordv, bordh, bordh, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        #plt.figure()
        #plt.imshow(imgr)
        #plt.show()
        #print(bordv, bordh)
        rm = cv2.getRotationMatrix2D((int(imgr.shape[0] / 2), int(imgr.shape[1] / 2)), gr, 1.0)

        imgr = cv2.warpAffine(imgr, rm, imgr.shape[1::-1], flags=cv2.INTER_LINEAR)
#         plt.figure()
#         plt.imshow(imgr)
#         plt.title(str(r['confidence']))
#         plt.show()
        # img_face = imgr        

        res2 = None
        try:
            boxes, conf, marks = mtcnn.detect(imgr, landmarks=True)  
            res2 = [{'box': boxes[i], 'confidence': conf[i], 
               'keypoints': marks[i]}  for i in range(len(boxes))]
        except Exception as ex:
            print("Erro em detect_faces", dy, dx, imgr.shape, r['confidence'])
#             cv2.imwrite("img_{:}_{:}_{:}_{:}.jpg".format(dy, dx, imgr.shape[0], imgr.shape[1]), imgr)
            print("Erro em detect_faces", ex)   
    
#             sign = -at/abs(at)
            rm = np.asarray([[math.cos(at), math.sin(at)],[-math.sin(at), math.cos(at)]])
            
            origin = np.asarray([x0n+bordv, y0n+bordh])
            rotorigin = rm @ origin[:, None].T[0] 
            kp = []
            for k, point in enumerate(r['keypoints']):
                point[0] = point[0] + bordv
                point[1] = point[1] + bordh
                
                p = (rm @ np.asarray(point)[:, None]).T[0]
#                 print(bordv, bordh, origin, point, rotorigin, p)
                p = [p[0] - rotorigin[0], p[1] - rotorigin[1]]
                kp.append(p)
    
            faces.append((imgr, coord, r['confidence'], kp))
            continue

        imgr_cy, imgr_cx = imgr.shape[0]/2, imgr.shape[1]/2

        fc2 = []
        min_dc = imgr_cy + imgr_cx
        min_img = None
        for r2 in res2:
            bb2 = r2['box']
            # coordenadas da detecção na imagem original
            
            x02 = max(bb2[0], 0)
            y02 = max(bb2[1], 0)
            x12 = min(bb2[2], pimg.shape[1])
            y12 = min(bb2[3], pimg.shape[0])
            
            w2, h2 = x12-x02, y12-y02
            
#             y02 = max(bb2[1], 0)
#             y12 = min(bb2[1] + bb2[3], imgr.shape[0])
#             x02 = max(bb2[0], 0)
#             x12 = min(bb2[0] + bb2[2], imgr.shape[1])
            dc = math.sqrt(((y02+y12)/2 - imgr_cy)**2 + ((x02+x12)/2 - imgr_cx)**2)

            # aumento da área detectada
            y0n2 = max(y02 - int(h2 * 0.08), 0)
            y1n2 = min(y12 + int(h2 * 0.08), imgr.shape[0])
            x0n2 = max(x02 - int(w2 * 0.08), 0)
            x1n2 = min(x12 + int(w2 * 0.08), imgr.shape[1])
            
            y0n2, y1n2, x0n2, x1n2 = int(y0n2), int(y1n2), int(x0n2), int(x1n2) 
            img_face = imgr[y0n2:y1n2, x0n2:x1n2]

            if dc < min_dc:
                min_dc = dc
                min_img = img_face
            #plt.figure()
            #plt.imshow(img_face)
            #plt.show()
            #faces.append((img_face, coord))

        if min_img is not None:
            #plt.figure()
            #plt.imshow(min_img)
            #plt.show()
            
            kp = []
            for k, point in enumerate(r2['keypoints']):
                p = [point[0] - x0n2, point[1] - y0n2]
                kp.append(p)
            
            faces.append((min_img, coord, r['confidence'], kp, rm))

    return faces

