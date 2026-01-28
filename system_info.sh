#!/bin/bash

# This script gathers system information in an HPC environment without sudo.

# --- Usage function ---
usage() {
    echo "Usage: $0 [-d|--destination <folder_path>]"
    echo "Options:"
    echo "  -d, --destination <folder_path>  Specify a destination folder to test its filesystem type and speed."
    exit 1
}

# --- Parse command-line arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--destination) DEST_FOLDER="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

echo "========================================================"
echo "          System Information          "
echo "========================================================"

# --- Operating System ---
echo ""
echo "--- Operating System ---"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "OS: $NAME"
    echo "Version: $VERSION"
else
    echo "Could not determine OS information from /etc/os-release."
fi

# --- CPU Information ---
echo ""
echo "--- CPU Information ---"
lscpu

# --- Memory Usage ---
echo ""
echo "--- Memory Usage ---"
free -h

# --- GPU Information ---
echo ""
echo "--- GPU Information ---"
echo "Detected VGA compatible controllers:"
lspci | grep -i vga
echo ""
echo "Attempting to get more detailed GPU info with nvidia-smi..."
if command -v nvidia-smi &> /dev/null
then
    nvidia-smi
else
    echo "nvidia-smi command not found. NVIDIA drivers may not be installed or in the PATH."
fi

# --- Disk Information ---
echo ""
echo "--- Disk Information ---"
echo "Block Devices:"
lsblk

echo ""
echo "Mounted Filesystems:"
df -hT

# --- Destination Folder Information ---
if [ -n "$DEST_FOLDER" ]; then
    echo ""
    echo "--- Destination Folder Information ---"
    if [ -d "$DEST_FOLDER" ]; then
        echo "Testing folder: $DEST_FOLDER"
        echo ""

        # --- Filesystem Type ---
        echo "Filesystem Type:"
        df -T "$DEST_FOLDER" | awk 'NR>1 {print $2}'
        echo ""

        # --- Disk Speed Test ---
        echo "--- Disk Speed (Attempt) ---"
        echo "NOTE: The following disk speed tests may not be accurate or may not work without sudo permissions."
        echo "These are best-effort attempts."
        
        TEMP_FILE="$DEST_FOLDER/tempfile_speedtest"

        # Write test
        echo ""
        echo "Attempting a basic write speed test with 'dd'..."
        dd if=/dev/zero of="$TEMP_FILE" bs=1G count=1 oflag=direct 2>&1 | grep "copied"

        # Read test
        echo ""
        echo "Attempting a basic read speed test with 'dd'..."
        dd if="$TEMP_FILE" of=/dev/null bs=1G count=1 iflag=direct 2>&1 | grep "copied"

        # Clean up the temp file
        rm -f "$TEMP_FILE"

    else
        echo "Error: Destination folder '$DEST_FOLDER' not found."
    fi
fi


echo ""
echo "For more accurate disk speed tests, 'hdparm' or 'fio' are recommended, but they typically require sudo."
echo "Example (if you had sudo): sudo hdparm -tT /dev/sda"
echo "Example (if you had sudo and fio installed): fio --name=randread --ioengine=libaio --iodepth=1 --rw=randread --bs=4k --size=1G --numjobs=1 --runtime=60 --group_reporting"


echo ""
echo "========================================================"
echo "          End of System Information"
echo "========================================================"
