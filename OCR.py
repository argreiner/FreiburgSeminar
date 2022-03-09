# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import cv2
import numpy as np
import pytesseract
from pandas import DataFrame

# +
pre = 'InhaltBL/BLVolIV'
post = '.png'
InhaltBL =''
for c in np.arange(1,3):
    file = pre+str(c)+post
    img_cv = cv2.imread(file,0)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    InhaltBL += pytesseract.image_to_string(img_rgb, lang='deu')

# By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
# we need to convert from BGR to RGB format/mode:
# img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
#print(pytesseract.image_to_string(img_rgb))
# OR
#img_rgb = Image.frombytes('RGB', img_cv.shape[:2], img_cv, 'raw', 'BGR', 0, 0)
#print(pytesseract.image_to_string(img_rgb))
# -

with open("InhaltBLVolIV.txt", "w") as text_file:
    text_file.write(InhaltBL)

InhaltBL

InhaltBLList = list(InhaltBL.split('\n'))

for zeile in InhaltBLList:
    print(zeile.count('.'))


