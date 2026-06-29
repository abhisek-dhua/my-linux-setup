#!/bin/bash

current_dir="/"

while true; do
    clear
    echo "=== Directory Size Analyzer ==="
    echo "Current directory: $current_dir"
    echo "Size: $(du -sh "$current_dir" 2>/dev/null | cut -f1)"
    echo ""
    
    if [[ "$current_dir" == "/" ]]; then
        echo "Top-level directories:"
    else
        echo "Contents of $current_dir:"
    fi
    
    echo ""
    echo "0) Go back (if not at root)"
    echo "q) Quit"
    echo ""
    
    # Get directory contents and display with numbers
    directories=()
    counter=1
    
    while IFS=$'\t' read -r size path; do
        if [[ -d "$path" && "$path" != "$current_dir" ]]; then
            display_name=$(basename "$path")
            echo "$counter) $size > $display_name"
            directories+=("$path")
            ((counter++))
        fi
    done < <(du -h --max-depth=1 "$current_dir" 2>/dev/null | sort -hr)
    
    echo ""
    read -p "Enter your choice: " choice
    
    if [[ "$choice" == "q" ]]; then
        echo "Exiting..."
        exit 0
    elif [[ "$choice" == "0" ]]; then
        if [[ "$current_dir" != "/" ]]; then
            current_dir=$(dirname "$current_dir")
            # Normalize to avoid trailing slash on root
            if [[ "$current_dir" == "" ]]; then
                current_dir="/"
            fi
        fi
    elif [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -lt "$counter" ]; then
        selected_dir="${directories[$((choice-1))]}"
        current_dir="$selected_dir"
    else
        echo "Invalid selection. Press Enter to continue..."
        read
    fi
done
