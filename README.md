# Temperature Transfer OpenCL Implementation
 Lightweight demo of how temperature changes from different cells to another in a 2D matrix with OpenCL

# Usage
 1. Install OpenCL dependencies
 2. Use an emulator for gcc, if on Windows, else install it on Linux
 3. Use the following syntax to compile: gcc _OpenCLUtil.c -o homework homework.c -L "<path_to_opencl_lib>\OCL_SDK_Light\lib\x86_64"  -I "<path_to_opencl_include_folder>\OCL_SDK_Light\include" -lOpenCL 
 4. Use the following syntax to run: homework.exe input.txt out.txt \<worker items> \<worker group size>
