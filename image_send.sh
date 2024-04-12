# sudo apt-get install inotify-tools

#!/bin/bash

WATCHED_DIR="/home/jhun/GAENet/input"

SERVER_URL="http://192.168.219.193:8000/upload-image/"

inotifywait -m -e create --format '%f' "${WATCHED_DIR}" | while read FILENAME
do
    echo "Detected new file: $FILENAME"
    FILE_EXTENSION="${FILENAME##*.}"

    if [[ $FILE_EXTENSION =~ ^(jpg|jpeg|png)$ ]]; then
        curl -X POST "${SERVER_URL}" \
             -F "file=@${WATCHED_DIR}/${FILENAME}" \
             -H "Content-Type: multipart/form-data"
    fi
done
