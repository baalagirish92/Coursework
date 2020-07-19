# We import Computer Vision and Pytesseract packages for optical character recognition.Tesseract - an open-source OCR engine developed intially by HP, now owned by Google.
# And Pytesseract is wrapper for the Tesseract OCR engine. The computer vision package OpenCV contains computer vision algorithms which can be used to identify texts in an
# image
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from pytesseract import Output

#This method uses the CV2 package to read the image and Pytesseract to print the text in the string
def image_processing(name, path):
    image=cv2.imread(path)  #image read using the imread method of Cv2
    text=pytesseract.image_to_string(image) #extracting text from images
    print("Text recognised from %s is : " %name)
    print(text) # print the text in the terminal
    d = pytesseract.image_to_data(image, output_type=Output.DICT) #get the data about the elements of the image- the text, positions of these words etc)
    print(d)
    n_boxes = len(d['text'])
    for i in range(n_boxes): 
        if int(d['conf'][i]) > 60: #take only those recognised characters/words with high confidence
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i]) #to obtain the coordinates of the word
            img = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) #create boxes around the words
            cv2.putText(image, d['text'][i], (x, y -35),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2) # put the recognised text over the box
            
    cv2.imshow(name, img) # display the image
    cv2.waitKey(0)



image_processing("dedication.jpg",r"F:\Semester3\KEDH\dedication.jpg")
image_processing("dh.jpg",r"F:\Semester3\KEDH\dh.jpg")




        

