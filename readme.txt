Contextual Edit Distance
------------------------

The main code is in the file CED_dolap.py, to run it, you have to : 

I]  COMPUTE CED FROM SCRATCH

    1)  Set up the three following parameters : alpha, f_k contextual function and the similariy.
        - f_k
        - sim
        - alpha

    2) Indicate the path of file operations in the FILE_PATH_OPERATIONS. 

    3) Comment the part "IMPORT CED DISTANCE MATRIX". 
    
    4) Uncomment the part "COMPUTE CED FROM SCRATCH". 


II]  IMPORT CED DISTANCE MATRIX

    1) Indicate the path of file CED matrix in the FILE_PATH_CED_DIST_MATRIX. 
    
    2) Comment the part "COMPUTE CED FROM SCRATCH"
    
    3) Uncomment the part "IMPORT CED DISTANCE MATRIX"
    
    
    
#####

About directories : 

* 'DATA-operations' contains all file used in the paper with the form 'xxx-operations.csv' : 
   - ipums-operations.csv 
   - artifical-operations.csv
   - open-operations.csv
   - security-operations.csv
   
* 'CED-matrices' contains all pre-computed CED distance matrices for each set of data 'ced-xxx.csv' : 
   - ced-ipums-.csv 
   - ced-artificial.csv
   - ced-open.csv
   - ced-security.csv
