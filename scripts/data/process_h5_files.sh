#!/bin/bash

# recursively go through all the h5 files in the current directory
for file in $(find ./multicoil_train -type f -name "*.h5"); do
    echo "Processing file: $file"
    # put the file into a directory named "RAW" in a directory from its name without the extension
    dirname=$(basename "$file")
    dirname="${dirname%.*}"
    mkdir -p "./multicoil_train/$dirname/RAW"
    mv "$file" "./multicoil_train/$dirname/RAW/"
    
done
