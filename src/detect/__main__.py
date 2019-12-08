import cv2
import numpy as np
import os
from PIL import Image
import pickle
# from sklearn.ensemble import RandomForestClassifier
import pandas as pd
# import xlsxwriter


default_image_size = tuple((45, 45))


def get_image_matrix(image, idx):
    try:
        image_grayscale = image.convert('L')
        # resize image
        image_grayscale = image_grayscale.resize(
            default_image_size, Image.ANTIALIAS)
        # image_grayscale.save('./Cropped_Element/'+'f'+str(idx)+'image.png', "png")
        image_np = np.array(image_grayscale)
        img_list = []
        for line in image_np:
            for value in line:
                img_list.append(value)
        return img_list
    except Exception as e:
        # print(image)
        return None

def  findx(item):
    item,_,_,_ = cv2.boundingRect(item)
    return item

def box_extraction(img_for_box_extraction_path, cropped_dir_path):

    img = cv2.imread(img_for_box_extraction_path, 1)  # Read the image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_blur = cv2.medianBlur(img,5).astype('uint8')
    (thresh, img_blur) = cv2.threshold(imgGray, 128, 255,
                                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the imagez
    img_bin = 255-img_blur  # Invert the image

    cv2.imwrite("Image_bin.jpg", img_bin)

    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//40
    
    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, kernel_length))
    # print(verticle_kernel)
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=2)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=2)
    cv2.imwrite("verticle_lines.jpg", verticle_lines_img)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=2)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=2)
    cv2.imwrite("horizontal_lines.jpg", horizontal_lines_img)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(
        verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(
        img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    cv2.imwrite("img_final_bin.jpg", img_final_bin)
    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(
        img_final_bin,  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    # (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")
    contours.reverse()

    idx = 18700
    z =0
    dx = -1
    dy = 0
    cXpre = 10000
    Arr = []
    r=0
    check = True
    check_arr = 0
    y_pre =0
    font = cv2.FONT_HERSHEY_SIMPLEX
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)
       
        # If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
       
        if (w > 30 and h > 15) and w > h:
            idx += 1
            z +=1
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # text = '{}+{}+{}'.format(z,cX, cY)
            # print('x =' ,cX, "  y=" ,cY)
            # cv2.putText(img_final_bin,text,(cX-80,cY), font, 1,(0,0,0),2,cv2.LINE_AA)
            while(check_arr <2 ):
                if(cY > y_pre):
                    check_arr+=1
                    y_pre +=50
            if cX-cXpre <0 :
                dx = dx +1
                Arr.append([])
                r = 0
                # print('cYpre =' ,cYpre, "  y=" ,cY)
            cXpre = cX  
            new_img = imgGray[y+3:y+h-3, x+3:x+w-3]
           
            # img_bin = 255-new_img
          
            im, thre = cv2.threshold(new_img, 170, 255, cv2.THRESH_BINARY_INV)
            cv2.imwrite('./Cropped/'+'a'+str(idx)+'.jpg', thre)
            contours_s, hierachy = cv2.findContours(
                thre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            rects =     [cv2.boundingRect(cnt) for cnt in contours_s]
            s = ''
            contours_s.sort(key=findx)

            a = []
            x_pre = None
            h_pre = None
            check = True
            path =''
            if contours_s is not None:
                while (check and (contours_s is not None)):
                    c = contours_s.pop(0)
                    (x, y, w, h) = cv2.boundingRect(c)
                    # print(x, y, w, h)
                    if x+y != 0: 
                        a.append(c)
                        x_pre = x
                        h_pre = h
                        check = False
                
                for i in contours_s:
                    (x, y, w, h) = cv2.boundingRect(i)
                    # print(x,y,w,h)
                    if (x+y!=0):
                        # print("x=",x,"  xpre=",x_pre)
                        # print("h=",h,"  hpre=",h_pre)

                        if x-x_pre <13:
                            if h>h_pre:
                                a.pop()
                                a.append(i)
                                # print(len(a))
                                x_pre = x
                                h_pre = h 
                                # print("x_pre:",x_pre,"h_pre",h_pre)
                                
                        else:
                            a.append(i)
                            x_pre = x
                            h_pre = h
                        
                # for r in a:
                #     print(cv2.boundingRect((r)))
                # print("There are : ",len(a) ," numbers")
                for con in a:
                    
                    (x, y, w, h) = cv2.boundingRect(con)
                    # print("Value ",x, y, w, h)
                    # print("contours is" ,i)
                    # if ((w > 10 and h > 15 and (x+y) > 0)) :
                    
                    # print("~~~~~~~~~~~~", idx)
                    if (w>10 and w<40 and h> 20 and y>0):
                        # print(x, y, w, h, idx)
                        idx = idx + 1
                        try:
                            
                            roi = new_img[y-2:y+h+2, x-2:x+w+2]
                            thre, im_bin_e = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY)
                            h,w = im_bin_e.shape[0],im_bin_e.shape[1]
                            wb = int ((h-w)/2)
                            impb = np.zeros((h,h))+255
                            impb= impb.astype('uint8')
                            impb[:,wb:wb+w]=im_bin_e
                            # path = './Cropped_Element/'+'g'+str(idx)+'.jpg'
                            # cv2.imwrite(path, impb)
                        except Exception as e:
                            try:
                                im_bin_e = new_img[y:y+h, x:x+w]
                                thre, im_bin_e = cv2.threshold(im_bin_e, 180, 255, cv2.THRESH_BINARY)
                                h,w = im_bin_e.shape[0],im_bin_e.shape[1]
                                wb = int ((h-w)/2)
                                impb = np.zeros((h,h))+255
                                impb= impb.astype('uint8')
                                impb[:,wb:wb+w]=im_bin_e
                                # path = './Cropped_Element/'+'g'+str(idx)+'.jpg'
                                # cv2.imwrite(path, impb)
                            except Exception as e:
                                im_bin_e = new_img[y:y+h, x:x+w]
                                thre, impb = cv2.threshold(im_bin_e, 180, 255, cv2.THRESH_BINARY)
                        
                        roi = Image.fromarray(impb)
                        image_file = [get_image_matrix(roi, idx)]
                        try:
                            saved_decision_tree_classifier_model = pickle.load(open("Model/random_forest_classifier.pkl",'rb'))
                            try:
                                model_prediction = saved_decision_tree_classifier_model.predict(image_file)
                                print(model_prediction)
                                # model_prediction = pytesseract.image_to_string(roi,config='--psm 10 --oem ')
                            except Exception as e:
                                model_prediction = ""
                                print("Can't model_prediction")
                            # print(model_prediction ," and ", w ,h)
                            st = model_prediction[0]
                            s = s + st
                            path = './Dataset/'+model_prediction[0]+'/'+'g'+str(idx)+'.jpg'
                           
                            cv2.imwrite(path, impb)
                            print(f"Recognized Digit : {model_prediction} ", path)
                        except FileNotFoundError as model_file_error:
                            print(f"Error : {model_file_error}")
    #                         print("... Training Model")
            r= r+1
            if r>1:
                if len(s)>1:
                    s =s[:1]+"."+s[1:]
            Arr[dx].append(s)
    print("Mang la:" ,Arr)
    Brr = []
    l =0
    for Ar in Arr:
        if len(Ar) >= l:
            l =len(Ar)
            Ar = Ar[:-1]
        Brr.append(Ar)

    cv2.imwrite("img_final_bin.jpg", img_final_bin)
    df = pd.DataFrame(data = Brr[3:])
    writer = pd.ExcelWriter('example.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1',header = False)
    writer.save()


box_extraction("0001.jpg", "./Cropped/")
