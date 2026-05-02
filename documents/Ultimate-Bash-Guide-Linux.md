# Ultimate Bash Guide for Linux

## Table of Contents

1. [Introduction to Bash](#introduction-to-bash)
2. [Bash Basics](#bash-basics)
3. [Variables and Data Types](#variables-and-data-types)
4. [Control Structures](#control-structures)
5. [Functions](#functions)
6. [File Operations](#file-operations)
7. [Text Processing](#text-processing)
8. [Command Substitution and Pipes](#command-substitution-and-pipes)
9. [Error Handling](#error-handling)
10. [Advanced Features](#advanced-features)
11. [Debugging and Testing](#debugging-and-testing)
12. [Best Practices](#best-practices)

---

## Introduction to Bash

### What is Bash?

Bash (Bourne Again Shell) is the default shell for most Linux distributions. It provides:

- **Command interpretation**: Execute commands and scripts
- **Scripting language**: Write automated tasks and programs
- **Environment management**: Handle variables and processes
- **Job control**: Manage background and foreground processes

### Bash vs Other Shells

- **sh**: Basic shell (Bourne shell)
- **bash**: Enhanced version of sh with additional features
- **zsh**: Advanced shell with many features
- **fish**: User-friendly shell with syntax highlighting

### Checking Your Shell

```bash
# Check current shell
echo $SHELL

# Check bash version
bash --version

# List available shells
cat /etc/shells

# Change default shell
chsh -s /bin/bash
```

---

## Bash Basics

### Script Structure

```bash
#!/bin/bash
# This is a comment
# Script name: example.sh
# Description: Basic bash script example

# Set strict error handling
set -euo pipefail

# Main script logic
echo "Hello, World!"
```

### Shebang Line

```bash
#!/bin/bash          # Use bash specifically
#!/usr/bin/env bash  # Use bash from PATH (more portable)
#!/bin/sh           # Use system default shell
```

### Making Scripts Executable

```bash
# Make script executable
chmod +x script.sh

# Run script
./script.sh

# Run with bash explicitly
bash script.sh

# Run with specific options
bash -x script.sh  # Debug mode
bash -n script.sh  # Syntax check only
```

### Basic Commands and Syntax

```bash
# Echo with options
echo "Hello World"
echo -e "Hello\nWorld"    # Interpret escape sequences
echo -n "No newline"      # Don't add newline

# Printf for formatted output
printf "Name: %s, Age: %d\n" "John" 25

# Read input
read -p "Enter your name: " name
read -s -p "Enter password: " password  # Silent input

# Variables
name="John"
age=25
echo "Name: $name, Age: $age"
```

---

## Variables and Data Types

### Variable Declaration and Assignment

```bash
# Basic variable assignment
name="John Doe"
age=25
price=19.99

# Read-only variables
readonly PI=3.14159

# Local variables (in functions)
local local_var="value"

# Array variables
fruits=("apple" "banana" "orange")
numbers=(1 2 3 4 5)

# Associative arrays (bash 4+)
declare -A person
person["name"]="John"
person["age"]="25"
```

### Variable Expansion

```bash
# Basic expansion
echo $name
echo ${name}

# Default values
echo ${name:-"Unknown"}
echo ${age:=0}

# String length
echo ${#name}

# Substring extraction
echo ${name:0:4}    # First 4 characters
echo ${name:5}      # From position 5 to end

# Pattern matching
filename="document.txt"
echo ${filename%.*}    # Remove extension
echo ${filename#*.}    # Get extension
echo ${filename/old/new}  # Replace first occurrence
echo ${filename//old/new} # Replace all occurrences
```

### Environment Variables

```bash
# Common environment variables
echo $HOME
echo $USER
echo $PATH
echo $PWD
echo $SHELL

# Set environment variable
export DATABASE_URL="mysql://localhost:3306/mydb"

# Unset variable
unset DATABASE_URL

# List all environment variables
env
printenv
```

### Special Variables

```bash
# Script parameters
echo $0    # Script name
echo $1    # First argument
echo $2    # Second argument
echo $#    # Number of arguments
echo $@    # All arguments as separate strings
echo $*    # All arguments as single string

# Process information
echo $$    # Current process ID
echo $!    # Last background process ID
echo $?    # Exit status of last command

# Example usage
#!/bin/bash
echo "Script: $0"
echo "Arguments: $#"
echo "First arg: $1"
echo "All args: $@"
```

---

## Control Structures

### Conditional Statements

#### if/elif/else

```bash
# Basic if statement
if [ $age -ge 18 ]; then
    echo "Adult"
fi

# if/else
if [ $age -ge 18 ]; then
    echo "Adult"
else
    echo "Minor"
fi

# if/elif/else
if [ $age -lt 13 ]; then
    echo "Child"
elif [ $age -lt 18 ]; then
    echo "Teenager"
else
    echo "Adult"
fi

# Using test command
if test $age -ge 18; then
    echo "Adult"
fi
```

#### Test Operators

```bash
# String comparisons
[ "$str1" = "$str2" ]     # Equal
[ "$str1" != "$str2" ]    # Not equal
[ -z "$str" ]             # Empty string
[ -n "$str" ]             # Non-empty string

# Numeric comparisons
[ $num1 -eq $num2 ]       # Equal
[ $num1 -ne $num2 ]       # Not equal
[ $num1 -lt $num2 ]       # Less than
[ $num1 -le $num2 ]       # Less than or equal
[ $num1 -gt $num2 ]       # Greater than
[ $num1 -ge $num2 ]       # Greater than or equal

# File tests
[ -f "$file" ]            # Regular file exists
[ -d "$dir" ]             # Directory exists
[ -r "$file" ]            # File is readable
[ -w "$file" ]            # File is writable
[ -x "$file" ]            # File is executable
[ -s "$file" ]            # File size > 0
[ -e "$file" ]            # File exists
```

#### Advanced Conditionals

```bash
# Logical operators
[ $age -ge 18 ] && [ $age -le 65 ]    # AND
[ $age -lt 18 ] || [ $age -gt 65 ]    # OR
! [ $age -lt 18 ]                     # NOT

# Compound conditions
if [[ $age -ge 18 && $age -le 65 ]]; then
    echo "Working age"
fi

# String pattern matching
if [[ "$filename" == *.txt ]]; then
    echo "Text file"
fi

# Regular expressions
if [[ "$email" =~ ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$ ]]; then
    echo "Valid email"
fi
```

### Loops

#### for Loop

```bash
# Basic for loop
for i in 1 2 3 4 5; do
    echo "Number: $i"
done

# Range-based for loop
for i in {1..5}; do
    echo "Number: $i"
done

# C-style for loop
for ((i=0; i<5; i++)); do
    echo "Number: $i"
done

# For loop with step
for i in {0..10..2}; do
    echo "Even number: $i"
done

# For loop with files
for file in *.txt; do
    echo "Processing: $file"
done

# For loop with command output
for user in $(cat users.txt); do
    echo "User: $user"
done
```

#### while Loop

```bash
# Basic while loop
counter=0
while [ $counter -lt 5 ]; do
    echo "Counter: $counter"
    ((counter++))
done

# While loop with read
while read line; do
    echo "Line: $line"
done < input.txt

# Infinite loop with break
while true; do
    echo "Press Ctrl+C to exit"
    sleep 1
done

# While loop with continue
counter=0
while [ $counter -lt 10 ]; do
    ((counter++))
    if [ $counter -eq 5 ]; then
        continue
    fi
    echo "Counter: $counter"
done
```

#### until Loop

```bash
# Until loop (opposite of while)
counter=0
until [ $counter -ge 5 ]; do
    echo "Counter: $counter"
    ((counter++))
done
```

#### Loop Control

```bash
# Break statement
for i in {1..10}; do
    if [ $i -eq 5 ]; then
        break
    fi
    echo "Number: $i"
done

# Continue statement
for i in {1..10}; do
    if [ $i -eq 5 ]; then
        continue
    fi
    echo "Number: $i"
done

# Nested loops with labels
outer_loop: for i in {1..3}; do
    for j in {1..3}; do
        if [ $i -eq 2 ] && [ $j -eq 2 ]; then
            break outer_loop
        fi
        echo "i=$i, j=$j"
    done
done
```

### Case Statement

```bash
# Basic case statement
case $day in
    "Monday")
        echo "Start of work week"
        ;;
    "Tuesday"|"Wednesday"|"Thursday")
        echo "Mid week"
        ;;
    "Friday")
        echo "End of work week"
        ;;
    "Saturday"|"Sunday")
        echo "Weekend"
        ;;
    *)
        echo "Unknown day"
        ;;
esac

# Case with patterns
case $filename in
    *.txt)
        echo "Text file"
        ;;
    *.jpg|*.jpeg|*.png)
        echo "Image file"
        ;;
    *.sh)
        echo "Shell script"
        ;;
    *)
        echo "Unknown file type"
        ;;
esac
```

---

## Functions

### Function Definition and Usage

```bash
# Basic function
function greet() {
    echo "Hello, $1!"
}

# Alternative syntax
greet() {
    echo "Hello, $1!"
}

# Function call
greet "John"

# Function with multiple parameters
add() {
    local result=$(( $1 + $2 ))
    echo $result
}

sum=$(add 5 3)
echo "Sum: $sum"

# Function with return value
multiply() {
    local result=$(( $1 * $2 ))
    return $result
}

multiply 4 5
echo "Return value: $?"
```

### Function Scope and Variables

```bash
# Global variable
global_var="I'm global"

# Function with local variable
test_function() {
    local local_var="I'm local"
    echo "Local: $local_var"
    echo "Global: $global_var"
}

test_function
echo "Outside local: $local_var"  # Will be empty
echo "Outside global: $global_var"  # Will work
```

### Function Parameters

```bash
# Accessing parameters
function process_args() {
    echo "Number of arguments: $#"
    echo "First argument: $1"
    echo "Second argument: $2"
    echo "All arguments: $@"
}

process_args "arg1" "arg2" "arg3"

# Function with default values
greet_with_default() {
    local name=${1:-"World"}
    echo "Hello, $name!"
}

greet_with_default        # Uses default
greet_with_default "John" # Uses provided value
```

### Recursive Functions

```bash
# Factorial function
factorial() {
    local n=$1
    if [ $n -le 1 ]; then
        echo 1
    else
        local prev=$(factorial $((n-1)))
        echo $((n * prev))
    fi
}

result=$(factorial 5)
echo "5! = $result"
```

---

## File Operations

### File Reading and Writing

```bash
# Read file line by line
while IFS= read -r line; do
    echo "Line: $line"
done < "input.txt"

# Read file into array
mapfile -t lines < "input.txt"
for line in "${lines[@]}"; do
    echo "Line: $line"
done

# Write to file
echo "Hello World" > output.txt
echo "Second line" >> output.txt

# Write multiple lines
cat > config.txt << EOF
server=localhost
port=8080
database=mydb
EOF

# Append to file
cat >> log.txt << EOF
$(date): New entry
EOF
```

### File Manipulation

```bash
# Check if file exists
if [ -f "file.txt" ]; then
    echo "File exists"
fi

# Create backup
cp original.txt original.txt.backup

# Safe file operations
if [ -f "important.txt" ]; then
    cp important.txt important.txt.backup
    echo "Backup created"
else
    echo "File not found"
fi

# File permissions
chmod 644 file.txt
chmod +x script.sh

# File ownership
chown user:group file.txt
```

### Directory Operations

```bash
# Create directory
mkdir new_directory
mkdir -p parent/child/grandchild  # Create nested directories

# Check if directory exists
if [ -d "directory" ]; then
    echo "Directory exists"
fi

# List directory contents
for item in *; do
    if [ -d "$item" ]; then
        echo "Directory: $item"
    else
        echo "File: $item"
    fi
done

# Find files
find . -name "*.txt" -type f
find . -mtime -7 -type f  # Modified in last 7 days
```

---

## Text Processing

### String Manipulation

```bash
# String length
str="Hello World"
echo ${#str}

# Substring
echo ${str:0:5}    # First 5 characters
echo ${str:6}      # From position 6 to end

# String replacement
echo ${str/World/Linux}     # Replace first occurrence
echo ${str//o/0}            # Replace all occurrences

# String removal
echo ${str#Hello}           # Remove prefix
echo ${str%World}           # Remove suffix

# Case conversion
echo ${str^^}               # Uppercase
echo ${str,,}               # Lowercase
echo ${str^}                # Capitalize first letter
```

### Text Processing with External Commands

```bash
# grep for pattern matching
grep "pattern" file.txt
grep -i "pattern" file.txt  # Case insensitive
grep -v "pattern" file.txt  # Invert match
grep -n "pattern" file.txt  # Show line numbers

# sed for text substitution
sed 's/old/new/g' file.txt
sed 's/old/new/' file.txt   # Replace first occurrence only
sed -i 's/old/new/g' file.txt  # Edit file in place

# awk for field processing
awk '{print $1}' file.txt   # Print first field
awk -F',' '{print $2}' file.txt  # Use comma as delimiter
awk '$1 > 10 {print $0}' file.txt  # Conditional printing

# cut for field extraction
cut -d',' -f1,3 file.txt    # Extract fields 1 and 3
cut -c1-10 file.txt         # Extract characters 1-10
```

### Regular Expressions

```bash
# Basic regex matching
if [[ "hello123" =~ [0-9]+ ]]; then
    echo "Contains numbers"
fi

# Email validation
email="user@example.com"
if [[ $email =~ ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$ ]]; then
    echo "Valid email"
fi

# Phone number validation
phone="123-456-7890"
if [[ $phone =~ ^[0-9]{3}-[0-9]{3}-[0-9]{4}$ ]]; then
    echo "Valid phone number"
fi
```

---

## Command Substitution and Pipes

### Command Substitution

```bash
# Basic command substitution
current_date=$(date)
echo "Current date: $current_date"

# Alternative syntax
current_date=`date`
echo "Current date: $current_date"

# Nested command substitution
file_count=$(ls -1 | wc -l)
echo "Number of files: $file_count"

# Command substitution in loops
for file in $(ls *.txt); do
    echo "Processing: $file"
done
```

### Pipes and Redirection

```bash
# Basic pipe
ls -la | grep "\.txt$"

# Multiple pipes
ps aux | grep "nginx" | awk '{print $2}' | xargs kill

# Input redirection
while read line; do
    echo "Processing: $line"
done < input.txt

# Output redirection
echo "Hello" > output.txt
echo "World" >> output.txt

# Error redirection
command 2> error.log
command 2>&1 | tee output.log

# Here documents
cat > config.txt << 'EOF'
server=localhost
port=8080
EOF
```

### Process Substitution

```bash
# Compare two files
diff <(sort file1.txt) <(sort file2.txt)

# Process multiple files
while read line; do
    echo "Processing: $line"
done < <(find . -name "*.txt")

# Combine commands
paste <(cut -d',' -f1 file1.txt) <(cut -d',' -f2 file2.txt)
```

---

## Error Handling

### Exit Codes

```bash
# Check exit status
command
if [ $? -eq 0 ]; then
    echo "Command succeeded"
else
    echo "Command failed"
fi

# Short form
if command; then
    echo "Command succeeded"
else
    echo "Command failed"
fi

# Multiple commands
command1 && command2 && command3
command1 || command2  # Run command2 only if command1 fails
```

### Error Handling with set

```bash
# Exit on error
set -e

# Exit on undefined variable
set -u

# Exit on pipe failure
set -o pipefail

# Combined (recommended)
set -euo pipefail

# Disable error exit
set +e
```

### Trap for Cleanup

```bash
#!/bin/bash

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    rm -f temp_file.txt
    exit 1
}

# Set trap
trap cleanup EXIT

# Script logic
echo "Creating temp file..."
echo "data" > temp_file.txt

# Normal exit
echo "Script completed successfully"
exit 0
```

### Custom Error Functions

```bash
# Error handling function
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# Usage
if [ ! -f "required_file.txt" ]; then
    error_exit "Required file not found"
fi

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> script.log
}

log "Script started"
```

---

## Advanced Features

### Arrays

```bash
# Declare array
declare -a fruits=("apple" "banana" "orange")

# Access elements
echo "${fruits[0]}"    # First element
echo "${fruits[@]}"    # All elements
echo "${#fruits[@]}"   # Array length

# Add elements
fruits+=("grape")

# Remove elements
unset fruits[1]

# Iterate over array
for fruit in "${fruits[@]}"; do
    echo "Fruit: $fruit"
done

# Associative arrays (bash 4+)
declare -A person
person["name"]="John"
person["age"]="25"
person["city"]="New York"

for key in "${!person[@]}"; do
    echo "$key: ${person[$key]}"
done
```

### Here Documents and Here Strings

```bash
# Here document
cat > config.txt << 'EOF'
server=localhost
port=8080
database=mydb
EOF

# Here string
grep "pattern" <<< "This is a test string"

# Here document with variables
name="John"
cat > greeting.txt << EOF
Hello, $name!
Welcome to our system.
EOF
```

### Parameter Expansion

```bash
# Default values
echo ${var:-"default"}
echo ${var:="default"}

# Error if unset
echo ${var:?"Variable is required"}

# Remove pattern from beginning
echo ${var#prefix}
echo ${var##prefix}  # Longest match

# Remove pattern from end
echo ${var%suffix}
echo ${var%%suffix}  # Longest match

# Replace pattern
echo ${var/old/new}
echo ${var//old/new}  # Global replacement

# Case conversion
echo ${var^}   # Capitalize first
echo ${var^^}  # All uppercase
echo ${var,}   # Lowercase first
echo ${var,,}  # All lowercase
```

### Subshells and Background Jobs

```bash
# Subshell
(cd /tmp && ls -la)

# Background job
sleep 10 &

# Wait for background job
wait $!

# Job control
jobs
fg %1
bg %1
kill %1
```

---

## Debugging and Testing

### Debugging Options

```bash
#!/bin/bash

# Enable debugging
set -x

# Debug specific section
set -x
# ... code to debug ...
set +x

# Run script with debug
bash -x script.sh

# Debug with line numbers
bash -v script.sh

# Syntax check only
bash -n script.sh
```

### Logging and Debugging

```bash
#!/bin/bash

# Debug function
debug() {
    if [ "$DEBUG" = "true" ]; then
        echo "DEBUG: $1" >&2
    fi
}

# Usage
DEBUG=true
debug "Starting script"
debug "Processing file: $filename"
```

### Testing Scripts

```bash
#!/bin/bash

# Test function
test_function() {
    local input="$1"
    local expected="$2"
    local result=$(process_input "$input")

    if [ "$result" = "$expected" ]; then
        echo "PASS: $input"
    else
        echo "FAIL: $input (expected: $expected, got: $result)"
    fi
}

# Run tests
test_function "input1" "expected1"
test_function "input2" "expected2"
```

---

## Best Practices

### Script Structure

```bash
#!/bin/bash
#
# Script Name: example.sh
# Description: Brief description of what the script does
# Author: Your Name
# Date: 2024-01-01
# Version: 1.0
#

# Set strict error handling
set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_FILE="$SCRIPT_DIR/script.log"

# Functions
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

error_exit() {
    log "ERROR: $1"
    exit 1
}

# Main function
main() {
    log "Script started"

    # Check prerequisites
    if [ ! -f "required_file.txt" ]; then
        error_exit "Required file not found"
    fi

    # Main logic
    log "Processing..."

    log "Script completed successfully"
}

# Run main function
main "$@"
```

### Security Best Practices

```bash
# Use quotes around variables
echo "$filename"  # Good
echo $filename    # Bad (can break with spaces)

# Use local variables in functions
function test_func() {
    local var="value"  # Good
    var="value"        # Bad (global)
}

# Validate input
if [ -z "$1" ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

# Use absolute paths when possible
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
```

### Performance Tips

```bash
# Use built-in commands over external
# Good
echo "${var#prefix}"
# Bad
echo "$var" | sed 's/^prefix//'

# Use arrays for multiple values
# Good
files=("file1.txt" "file2.txt")
for file in "${files[@]}"; do
    process "$file"
done
# Bad
for file in file1.txt file2.txt; do
    process "$file"
done

# Avoid subshells when possible
# Good
readarray -t lines < file.txt
# Bad
lines=($(cat file.txt))
```

### Portability

```bash
#!/bin/bash
# Use shebang with env for portability
#!/usr/bin/env bash

# Check for required commands
command -v jq >/dev/null 2>&1 || { echo "jq is required but not installed. Aborting." >&2; exit 1; }

# Use POSIX-compliant syntax when possible
# Good
[ "$var" = "value" ]
# Bad
[[ "$var" == "value" ]]
```

---

## Quick Reference

### Essential Commands

```bash
# Variables
var="value"              # Assignment
echo "$var"              # Expansion
${var:-default}          # Default value

# Conditionals
if [ condition ]; then   # if statement
case $var in             # case statement
for item in list; do     # for loop
while [ condition ]; do  # while loop

# Functions
function_name() {        # Function definition
return $value            # Return value

# File operations
[ -f "$file" ]           # File exists
[ -d "$dir" ]            # Directory exists
read -r line < file      # Read file
echo "text" > file       # Write file
```

### Common Patterns

```bash
# Check if command exists
command -v cmd >/dev/null 2>&1

# Check if file exists
[ -f "$file" ]

# Check if variable is set
[ -n "$var" ]

# Loop through files
for file in *.txt; do

# Read file line by line
while IFS= read -r line; do

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
```

This comprehensive bash guide covers all essential aspects of bash scripting on Linux systems. Remember to always test your scripts thoroughly and follow security best practices.
