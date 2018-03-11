# coding:utf-8
import sys
import os
import re
import shutil
notebook = sys.argv[1]
texFile = notebook.replace(��.ipynb��,��.tex��)
# 1.convert .ipynb to latex file .tex
# ��ipynb�ļ�תΪtex�ļ�
print ��1. convert �� + notebook + �� to �� + texFile  
print ��------ \n��
os.system(r��jupyter nbconvert --to latex �� + notebook)
print ��convert over��
# 2. add Chinese support by adding the string below
# �������õİ�ʹ֧�����ģ�ֱ��ת�����Ļᶪʧ��
# \usepackage{fontspec, xunicode, xltxtra}
# \setmainfont{Microsoft YaHei} 
# \usepackage{ctex}
print ��2. add Chinese support to .tex file�� 
print ��------��
file = open(texFile,��r��)
str_file = file.read()
strinfo = re.compile(��(documentclass[\d\D]+\{article\})��)  #���ҵ��ַ�line0
m=re.findall(strinfo,str_file)
if len(m) == 0:
    print r��can not find documentclass[**pt]{article}��
    sys.exit(1)
str_file = strinfo.sub(��\\1 \n \\usepackage{fontspec, xunicode, xltxtra} \n \\setmainfont{Microsoft YaHei} \r \\usepackage{ctex}��,str_file)  #  �滻���ַ�line1
file.close()
file = open(texFile,��w��)
file.write(str_file)
file.close()
print ��add Chinese support successed��
# 3. convert .tex to .pdf by xelatex
# ʹ��xelatex�������.tex�ļ��õ�pdf
print  ��3. convert tex to pdf��
print ��------��
os.system(��xelatex �� + texFile)
print  ��convert pdf successed��
# 4. delete the auxiliary files
# �������ɵ��м��ļ�
# change there if latex file is needed
print ��4. delete auxiliary files��
print ��------��
os.remove(notebook.replace(��.ipynb��,��.aux��))
os.remove(notebook.replace(��.ipynb��,��.log��))
os.remove(notebook.replace(��.ipynb��,��.out��))
# change there if latex file is needed
os.remove(notebook.replace(��.ipynb��,��.tex��))
if os.path.isdir(notebook.replace(��.ipynb��,��_files��)):
    shutil.rmtree(notebook.replace(��.ipynb��,��_files��))
print ��delete auxiliary files successed��