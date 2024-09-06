IF "%~1"=="" (
    SET INPUT_FILE=.\main.hip
) ELSE (
    SET INPUT_FILE=%~1
)
IF "%~2"=="" (
    SET OUTPUT_FILE=a.exe
) ELSE (
    SET OUTPUT_FILE=%~2
)

hipcc %INPUT_FILE% -I "C:\ROCm\include" -L "C:\ROCm\lib" --offload-arch=gfx1030 -o %OUTPUT_FILE% -std=c++14 -O3