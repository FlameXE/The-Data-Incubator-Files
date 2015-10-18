set /p py_file = Enter the .py file: 

set /p content_file = Enter the content file: 

set /p output_file = Enter the output file: 

cls

call python %py_file % %content_file % > %output_file %.txt

cls

call %output_file %.txt