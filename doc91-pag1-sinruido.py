import os
import fitz
from PIL import Image
import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

def find_text_coordinates(gray_image):
    boxes = pytesseract.image_to_boxes(gray_image, lang='spa')
    coordinates = []
    for box in boxes.splitlines():
        box = box.split(' ')
        x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
        coordinates.append((x, y, w, h))
    return coordinates
def transformar_blancos_a_gris(image, valor_gris):
    umbral_blancos=20
    if len(image.shape) > 2:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image
    mask = np.logical_or(image_gray >= 255 - umbral_blancos, image_gray <= umbral_blancos)
    image_gray[mask] = valor_gris
    return image_gray
def create_circle_mask(gray_image, coordinates):
    mask = np.ones_like(gray_image, dtype=np.uint8) * 255
    for x, y, _, _ in coordinates:
        y = gray_image.shape[0] - y
        cv2.circle(mask, (x, y), 5, (0, 0, 255), -1)
    return mask
def find_largest_contour(contours):
    max_area = 0
    largest_contour = None
    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)
        if h<100 or w<100:
            continue
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour
    return largest_contour
def desenfoque_objetos(image,ite):
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(image,150,255,cv2.THRESH_BINARY_INV) # threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilated = cv2.dilate(thresh,kernel,iterations = ite) # dilate
    contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours
    return contours,hierarchy
def comprobar(image,ite):
    cont=0
    contours,_=desenfoque_objetos(image,ite)
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        if h>100:
            cont+=1
    if(cont>10):
        imagen_girada = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) 
        return imagen_girada
    return image
def corte_palabras(nombre_documento,image,ite,h_value_min,w_value_min,h_value_max,w_value_max,contseg,page_number):
    cont=0
    contours,_=desenfoque_objetos(image,ite)
    #text_boxes = []
    contours_to_reprocess = []
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        if h < h_value_min or w < w_value_min:
            continue
        if h > h_value_max or w > w_value_max:
            #contours_to_reprocess.append(contour)
            continue
        roi_image = Image.fromarray(image)
        cropped_img = roi_image.crop((x, y, x + w, y + h))
        text = pytesseract.image_to_string(cropped_img, lang='spa')
        if len(text.strip()) >= 0:  
            cropped_np = np.array(cropped_img)
            blurred_image = cv2.GaussianBlur(cropped_np, (7, 7), 0)
            binarized_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 3)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opened_image = cv2.morphologyEx(binarized_image, cv2.MORPH_OPEN, kernel)
            cv2.imwrite(f'{nombre_documento}-{page_number}_img{x}_{y}.png', opened_image)
            image[y:y + h, x:x + w] = 207
            cont+=1
    if(cont<=10):
        contseg+=1
        if(contseg<=4):
           corte_palabras(nombre_documento,image,ite-1,h_value_min,w_value_min,h_value_max,w_value_max,contseg,page_number)
def primer_filtro(image,nombre_documento,page_number):
    m=0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    custom_config = r'--psm 6'
    results = pytesseract.image_to_data(image, lang='spa+spa_old', config=custom_config, output_type=pytesseract.Output.DICT)
    for i, word_text in enumerate(results['text']):
        if word_text.strip():
            x, y, w, h = results['left'][i], results['top'][i], results['width'][i], results['height'][i]
            cropped_img = image[y:y+h, x:x+w]
            text = pytesseract.image_to_string(cropped_img,lang='spa+spa_old')
            if w<15 or h<15:
                continue
            if(len(text)>=1):
                cropped_img_gray = gray_image[y:y+h, x:x+w]
                blurred_image = cv2.GaussianBlur(cropped_img_gray, (7, 7), 0)
                binarized_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
                cv2.imwrite(f'doc_{nombre_documento}_pag{page_number}_img{m}.png', binarized_image)
                gray_image[y+5:y + h-5, x+5:x + w-5] = 207
                m+=1
    return gray_image

def submain(nombre_documento,image,page_number):
    gray_image=primer_filtro(image,nombre_documento,page_number)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image=transformar_blancos_a_gris(gray_image,207)
    gray_image=comprobar(gray_image,7)
    text_coordinates = find_text_coordinates(gray_image)
    circle_mask = create_circle_mask(gray_image, text_coordinates)
    contours,_=desenfoque_objetos(circle_mask,28)
    largest_contour = find_largest_contour(contours)

    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        roi_image = Image.fromarray(gray_image)
        cropped_img = roi_image.crop((x, y, x + w, y + h))
        cropped_np = np.array(cropped_img)
        

        corte_palabras(nombre_documento,cropped_np,7,25,25,80,200,0,page_number)
        gray_image[y:y + h, x:x + w] = 207
    corte_palabras(nombre_documento,gray_image,7,30,30,80,200,4,page_number)

# Ruta al archivo PDF que deseas procesar
pdf_path = r'C:\Users\User\OneDrive - UNIVERSIDAD NACIONAL DE INGENIERIA\Documents\ONELOOP-START UP\PROYECTOS\PROYECTO1-HTR\91.pdf'
output_directory = r'C:\Users\User\OneDrive - UNIVERSIDAD NACIONAL DE INGENIERIA\Documents\ONELOOP-START UP\PROYECTOS\PROYECTO1-HTR'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)
pdf_document = fitz.open(pdf_path)

for page_number in range(pdf_document.page_count):

    page = pdf_document[page_number]
    output_file = os.path.join(output_directory, f'page_{page_number + 1}.jpg')

    img = page.get_pixmap(matrix=fitz.Matrix(300 / 144, 300 / 144))  # 150 dpi
    img = Image.frombytes("RGB", [img.width, img.height], img.samples)
    img_np = np.array(img)
    
    submain(91,img_np,page_number)
    if(page_number==0):
        break

pdf_document.close()
print('Proceso completado.')