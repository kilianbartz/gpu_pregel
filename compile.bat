IF "%~1"=="" (
    SET INPUT_FILE=.\main.hip
) ELSE (
    SET INPUT_FILE=%~1
)

hipcc %INPUT_FILE% -I "C:\ROCm\include" -L "C:\ROCm\lib" --offload-arch=gfx1030