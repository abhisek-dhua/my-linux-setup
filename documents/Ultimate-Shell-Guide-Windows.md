# Ultimate Shell Guide for Windows

## Table of Contents

1. [Introduction to Windows Shells](#introduction-to-windows-shells)
2. [PowerShell Basics](#powershell-basics)
3. [PowerShell Scripting](#powershell-scripting)
4. [Command Prompt (CMD)](#command-prompt-cmd)
5. [Batch Scripting](#batch-scripting)
6. [Cross-Platform Solutions](#cross-platform-solutions)
7. [Windows Subsystem for Linux (WSL)](#windows-subsystem-for-linux-wsl)
8. [Advanced Features](#advanced-features)
9. [Error Handling and Debugging](#error-handling-and-debugging)
10. [Best Practices](#best-practices)

---

## Introduction to Windows Shells

### Available Shells on Windows

- **Command Prompt (CMD)**: Traditional Windows command shell
- **PowerShell**: Modern scripting and automation platform
- **PowerShell Core**: Cross-platform version of PowerShell
- **Windows Subsystem for Linux (WSL)**: Linux environment on Windows
- **Git Bash**: Unix-like shell for Git operations

### Choosing the Right Shell

- **PowerShell**: Modern Windows administration and scripting
- **CMD**: Legacy scripts and simple commands
- **WSL**: Linux tools and scripts
- **Git Bash**: Git operations and Unix tools

### Checking Available Shells

```powershell
# Check PowerShell version
$PSVersionTable

# Check if WSL is available
wsl --list --verbose

# Check Git Bash
where bash
```

---

## PowerShell Basics

### PowerShell Execution Policy

```powershell
# Check current execution policy
Get-ExecutionPolicy

# Set execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser

# Available policies:
# - Restricted: No scripts allowed
# - AllSigned: Only signed scripts
# - RemoteSigned: Local scripts + signed remote scripts
# - Unrestricted: All scripts allowed
```

### Basic PowerShell Commands

```powershell
# Get help
Get-Help Get-Process
Get-Help about_*

# Get command information
Get-Command Get-Process
Get-Command -CommandType Cmdlet

# List available commands
Get-Command | Where-Object {$_.Name -like "*Process*"}

# Get system information
Get-ComputerInfo
Get-Process
Get-Service
Get-EventLog -LogName Application -Newest 10
```

### Variables and Data Types

```powershell
# Variable declaration
$name = "John Doe"
$age = 25
$price = 19.99
$isActive = $true

# Strongly typed variables
[int]$number = 42
[string]$text = "Hello"
[datetime]$date = Get-Date

# Arrays
$fruits = @("apple", "banana", "orange")
$numbers = @(1, 2, 3, 4, 5)

# Hash tables (associative arrays)
$person = @{
    Name = "John"
    Age = 25
    City = "New York"
}

# Read-only variables
Set-Variable -Name "PI" -Value 3.14159 -Option ReadOnly
```

### String Operations

```powershell
# String concatenation
$firstName = "John"
$lastName = "Doe"
$fullName = "$firstName $lastName"

# String formatting
$message = "Hello, {0}! You are {1} years old." -f $name, $age

# String methods
$text = "Hello World"
$text.Length
$text.ToUpper()
$text.ToLower()
$text.Substring(0, 5)
$text.Replace("World", "PowerShell")

# Here strings
$multiline = @"
This is a
multiline string
with multiple lines
"@
```

---

## PowerShell Scripting

### Script Structure

```powershell
#Requires -Version 5.1
#Requires -RunAsAdministrator

<#
.SYNOPSIS
    Brief description of the script
.DESCRIPTION
    Detailed description of what the script does
.PARAMETER ParameterName
    Description of the parameter
.EXAMPLE
    .\script.ps1 -ParameterName "value"
.NOTES
    Additional notes and information
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ParameterName,

    [Parameter(Mandatory=$false)]
    [int]$Count = 10
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Main script logic
Write-Host "Script started with parameter: $ParameterName"

# Script body
for ($i = 1; $i -le $Count; $i++) {
    Write-Host "Processing item $i"
}

Write-Host "Script completed successfully"
```

### Control Structures

#### Conditional Statements

```powershell
# if/elseif/else
if ($age -ge 18) {
    Write-Host "Adult"
} elseif ($age -ge 13) {
    Write-Host "Teenager"
} else {
    Write-Host "Child"
}

# Switch statement
switch ($day) {
    "Monday" { Write-Host "Start of work week" }
    "Tuesday" { Write-Host "Mid week" }
    "Friday" { Write-Host "End of work week" }
    default { Write-Host "Unknown day" }
}

# Switch with wildcards
switch -Wildcard ($filename) {
    "*.txt" { Write-Host "Text file" }
    "*.jpg" { Write-Host "Image file" }
    "*.ps1" { Write-Host "PowerShell script" }
    default { Write-Host "Unknown file type" }
}
```

#### Loops

```powershell
# For loop
for ($i = 0; $i -lt 5; $i++) {
    Write-Host "Number: $i"
}

# ForEach loop
$fruits = @("apple", "banana", "orange")
foreach ($fruit in $fruits) {
    Write-Host "Fruit: $fruit"
}

# ForEach-Object (pipeline)
1..5 | ForEach-Object { Write-Host "Number: $_" }

# While loop
$counter = 0
while ($counter -lt 5) {
    Write-Host "Counter: $counter"
    $counter++
}

# Do-While loop
$counter = 0
do {
    Write-Host "Counter: $counter"
    $counter++
} while ($counter -lt 5)
```

### Functions

```powershell
# Basic function
function Get-Greeting {
    param([string]$Name)
    return "Hello, $Name!"
}

# Function with multiple parameters
function Add-Numbers {
    param(
        [int]$First,
        [int]$Second
    )
    return $First + $Second
}

# Function with default values
function Get-UserInfo {
    param(
        [string]$Username = $env:USERNAME,
        [switch]$Verbose
    )

    if ($Verbose) {
        Write-Host "Getting info for user: $Username"
    }

    return Get-ADUser -Identity $Username
}

# Advanced function with pipeline support
function Get-FileInfo {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true, ValueFromPipeline=$true)]
        [string]$Path
    )

    process {
        if (Test-Path $Path) {
            $file = Get-Item $Path
            [PSCustomObject]@{
                Name = $file.Name
                Size = $file.Length
                Modified = $file.LastWriteTime
            }
        }
    }
}

# Usage
Get-FileInfo -Path "C:\temp\file.txt"
"C:\temp\file1.txt", "C:\temp\file2.txt" | Get-FileInfo
```

### Error Handling

```powershell
# Try-Catch-Finally
try {
    $result = 10 / 0
} catch {
    Write-Host "Error occurred: $($_.Exception.Message)"
} finally {
    Write-Host "Cleanup code here"
}

# Error action preference
$ErrorActionPreference = "Stop"  # Stop on any error
$ErrorActionPreference = "Continue"  # Continue on errors
$ErrorActionPreference = "SilentlyContinue"  # Suppress errors

# Error handling with specific error types
try {
    Get-Content "nonexistent.txt"
} catch [System.IO.FileNotFoundException] {
    Write-Host "File not found"
} catch {
    Write-Host "Other error: $($_.Exception.Message)"
}
```

---

## Command Prompt (CMD)

### Basic CMD Commands

```cmd
REM Comments in batch files
:: Alternative comment syntax

REM Display help
help command
command /?

REM Directory operations
dir
cd directory
mkdir directory
rmdir directory

REM File operations
copy source dest
move source dest
del file
type file

REM System information
systeminfo
ver
hostname
whoami

REM Network commands
ipconfig
ping hostname
netstat -an
```

### Environment Variables

```cmd
REM Display environment variables
set
echo %PATH%

REM Set environment variable
set VARIABLE=value
setx VARIABLE "value"

REM Use environment variables
echo %USERNAME%
echo %COMPUTERNAME%
echo %TEMP%

REM Conditional with environment variables
if "%ERRORLEVEL%"=="0" (
    echo Command succeeded
) else (
    echo Command failed
)
```

### CMD Scripting Features

```cmd
REM Command chaining
command1 & command2
command1 && command2
command1 || command2

REM Input redirection
command < input.txt
command > output.txt
command >> output.txt

REM Pipes
command1 | command2

REM Background execution
start command
```

---

## Batch Scripting

### Batch Script Structure

```batch
@echo off
REM Batch script example
REM Author: Your Name
REM Date: 2024-01-01

setlocal enabledelayedexpansion

REM Set variables
set "SCRIPT_DIR=%~dp0"
set "LOG_FILE=%SCRIPT_DIR%script.log"

REM Main logic
echo Starting script...
call :log "Script started"

REM Process files
for %%f in (*.txt) do (
    echo Processing: %%f
    call :process_file "%%f"
)

call :log "Script completed"
goto :eof

REM Functions
:log
echo %date% %time% - %~1 >> "%LOG_FILE%"
goto :eof

:process_file
echo Processing file: %~1
goto :eof
```

### Variables and Parameters

```batch
REM Script parameters
echo Script name: %0
echo First parameter: %1
echo Second parameter: %2
echo All parameters: %*

REM Local variables
setlocal
set "local_var=value"
echo %local_var%
endlocal

REM Delayed expansion
setlocal enabledelayedexpansion
for %%i in (1 2 3) do (
    set "var=%%i"
    echo !var!
)
endlocal
```

### Control Structures

```batch
REM if/else statements
if exist "file.txt" (
    echo File exists
) else (
    echo File not found
)

if "%var%"=="value" (
    echo Variable equals value
) else (
    echo Variable does not equal value
)

REM for loops
for %%i in (1 2 3 4 5) do (
    echo Number: %%i
)

for %%f in (*.txt) do (
    echo File: %%f
)

for /f "tokens=1,2 delims=," %%a in (data.csv) do (
    echo Field1: %%a, Field2: %%b
)

REM while loop (using goto)
:loop
if %counter% lss 5 (
    echo Counter: %counter%
    set /a counter+=1
    goto loop
)
```

### Advanced Batch Features

```batch
REM Error handling
command 2>nul
if errorlevel 1 (
    echo Command failed
) else (
    echo Command succeeded
)

REM User input
set /p "user_input=Enter your name: "
echo Hello, %user_input%!

REM Date and time
echo Current date: %date%
echo Current time: %time%

REM Random numbers
set /a random_num=%random% %% 100
echo Random number: %random_num%

REM String manipulation
set "text=Hello World"
echo Original: %text%
echo Length: %text:~0,5%
echo Substring: %text:~6,5%
```

---

## Cross-Platform Solutions

### Python Scripts

```python
#!/usr/bin/env python3
"""
Cross-platform script example
Works on Windows, Linux, and macOS
"""

import os
import sys
import platform
import subprocess

def get_system_info():
    """Get system information"""
    return {
        'platform': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor()
    }

def run_command(command):
    """Run a command and return output"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return None

def main():
    """Main function"""
    print("Cross-platform script")

    # Get system info
    info = get_system_info()
    print(f"Platform: {info['platform']}")

    # Run platform-specific commands
    if info['platform'] == 'Windows':
        result = run_command('dir')
    else:
        result = run_command('ls -la')

    if result:
        print("Command output:")
        print(result)

if __name__ == "__main__":
    main()
```

### Node.js Scripts

```javascript
#!/usr/bin/env node
/**
 * Cross-platform Node.js script
 */

const os = require("os")
const { exec } = require("child_process")
const { promisify } = require("util")

const execAsync = promisify(exec)

async function getSystemInfo() {
  return {
    platform: os.platform(),
    hostname: os.hostname(),
    userInfo: os.userInfo(),
    totalMemory: os.totalmem(),
    freeMemory: os.freemem(),
  }
}

async function runCommand(command) {
  try {
    const { stdout, stderr } = await execAsync(command)
    return stdout.trim()
  } catch (error) {
    console.error(`Error running command: ${error.message}`)
    return null
  }
}

async function main() {
  console.log("Cross-platform Node.js script")

  // Get system info
  const info = await getSystemInfo()
  console.log(`Platform: ${info.platform}`)

  // Run platform-specific commands
  let result
  if (info.platform === "win32") {
    result = await runCommand("dir")
  } else {
    result = await runCommand("ls -la")
  }

  if (result) {
    console.log("Command output:")
    console.log(result)
  }
}

main().catch(console.error)
```

### Shell Scripts with Git Bash

```bash
#!/bin/bash
# Cross-platform shell script using Git Bash on Windows

# Detect operating system
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="Windows"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
else
    OS="Unknown"
fi

echo "Operating System: $OS"

# Platform-specific commands
case $OS in
    "Windows")
        echo "Running on Windows"
        # Use Git Bash commands
        ls -la
        ;;
    "Linux"|"macOS")
        echo "Running on Unix-like system"
        ls -la
        ;;
    *)
        echo "Unknown operating system"
        ;;
esac

# Cross-platform file operations
if [ -f "config.txt" ]; then
    echo "Config file exists"
    cat config.txt
else
    echo "Config file not found"
fi
```

---

## Windows Subsystem for Linux (WSL)

### Installing and Configuring WSL

```powershell
# Enable WSL feature
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux

# Install WSL 2
wsl --install

# List available distributions
wsl --list --online

# Install specific distribution
wsl --install -d Ubuntu
wsl --install -d Debian
wsl --install -d openSUSE-42

# Set WSL 2 as default
wsl --set-default-version 2

# List installed distributions
wsl --list --verbose
```

### Using WSL

```bash
# Access WSL from Windows
wsl

# Run specific command in WSL
wsl ls -la

# Access Windows files from WSL
ls /mnt/c/Users/username/Documents

# Access WSL files from Windows
# Files are located in: \\wsl$\Ubuntu\home\username

# Run Windows commands from WSL
cmd.exe /c dir
powershell.exe -Command "Get-Process"
```

### WSL Scripting

```bash
#!/bin/bash
# WSL script example

# Check if running in WSL
if grep -qi microsoft /proc/version; then
    echo "Running in WSL"
    WINDOWS_USER=$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d '\r')
    echo "Windows user: $WINDOWS_USER"
else
    echo "Running in native Linux"
fi

# Access Windows environment variables
WINDOWS_PATH=$(cmd.exe /c "echo %PATH%" 2>/dev/null | tr -d '\r')
echo "Windows PATH: $WINDOWS_PATH"

# Cross-platform file operations
WINDOWS_DOCS="/mnt/c/Users/$WINDOWS_USER/Documents"
if [ -d "$WINDOWS_DOCS" ]; then
    echo "Windows Documents directory: $WINDOWS_DOCS"
    ls "$WINDOWS_DOCS"
fi
```

---

## Advanced Features

### PowerShell Advanced Functions

```powershell
function Get-SystemReport {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter(Mandatory=$false)]
        [switch]$IncludeProcesses,

        [Parameter(Mandatory=$false)]
        [switch]$IncludeServices
    )

    begin {
        Write-Verbose "Starting system report generation"
    }

    process {
        $report = [PSCustomObject]@{
            ComputerName = $env:COMPUTERNAME
            OS = (Get-ComputerInfo).WindowsProductName
            Uptime = (Get-Date) - (Get-CimInstance Win32_OperatingSystem).LastBootUpTime
            Memory = Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory
        }

        if ($IncludeProcesses) {
            $report | Add-Member -MemberType NoteProperty -Name "TopProcesses" -Value (Get-Process | Sort-Object CPU -Descending | Select-Object -First 5)
        }

        if ($IncludeServices) {
            $report | Add-Member -MemberType NoteProperty -Name "Services" -Value (Get-Service | Where-Object {$_.Status -eq "Running"})
        }

        return $report
    }

    end {
        Write-Verbose "System report generation completed"
    }
}

# Usage
Get-SystemReport -IncludeProcesses -IncludeServices -Verbose
```

### PowerShell Modules

```powershell
# Create module directory
New-Item -ItemType Directory -Path "$env:USERPROFILE\Documents\WindowsPowerShell\Modules\MyModule" -Force

# Create module manifest
New-ModuleManifest -Path "$env:USERPROFILE\Documents\WindowsPowerShell\Modules\MyModule\MyModule.psd1" -RootModule "MyModule.psm1" -Author "Your Name" -Description "My custom module"

# Create module script
@"
function Get-CustomInfo {
    param([string]`$Name)
    return "Hello, `$Name!"
}

Export-ModuleMember -Function Get-CustomInfo
"@ | Out-File -FilePath "$env:USERPROFILE\Documents\WindowsPowerShell\Modules\MyModule\MyModule.psm1" -Encoding UTF8

# Import and use module
Import-Module MyModule
Get-CustomInfo "World"
```

### Advanced Batch Scripting

```batch
@echo off
setlocal enabledelayedexpansion

REM Advanced batch script with functions and error handling

REM Configuration
set "SCRIPT_NAME=%~n0"
set "SCRIPT_DIR=%~dp0"
set "LOG_FILE=%SCRIPT_DIR%%SCRIPT_NAME%.log"
set "ERROR_LOG=%SCRIPT_DIR%%SCRIPT_NAME%_error.log"

REM Initialize
call :log "Script started"
call :check_prerequisites

REM Main processing
for %%f in (*.txt) do (
    call :process_file "%%f"
    if errorlevel 1 (
        call :log "ERROR: Failed to process %%f"
    ) else (
        call :log "SUCCESS: Processed %%f"
    )
)

call :log "Script completed"
goto :eof

REM Function: Logging
:log
echo %date% %time% - %~1 >> "%LOG_FILE%"
goto :eof

REM Function: Check prerequisites
:check_prerequisites
if not exist "config.ini" (
    call :log "ERROR: config.ini not found"
    exit /b 1
)
goto :eof

REM Function: Process file
:process_file
set "file=%~1"
call :log "Processing: %file%"

REM Add your processing logic here
echo Processing %file%...

REM Simulate success/failure
set /a random=!random! %% 2
if !random! equ 0 (
    exit /b 1
) else (
    exit /b 0
)
```

---

## Error Handling and Debugging

### PowerShell Error Handling

```powershell
# Comprehensive error handling
function Invoke-SafeCommand {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)]
        [scriptblock]$ScriptBlock,

        [string]$ErrorMessage = "Command failed",
        [switch]$ContinueOnError
    )

    try {
        $result = & $ScriptBlock
        return $result
    } catch {
        $errorDetails = @{
            Message = $_.Exception.Message
            Type = $_.Exception.GetType().Name
            Line = $_.InvocationInfo.Line
            ScriptStackTrace = $_.ScriptStackTrace
        }

        Write-Error "$ErrorMessage`: $($errorDetails.Message)"
        Write-Debug "Error details: $($errorDetails | ConvertTo-Json)"

        if (-not $ContinueOnError) {
            throw
        }
    }
}

# Usage
Invoke-SafeCommand -ScriptBlock { Get-Content "nonexistent.txt" } -ErrorMessage "File read failed" -ContinueOnError
```

### Debugging Techniques

```powershell
# PowerShell debugging
Set-PSDebug -Trace 1  # Trace script execution
Set-PSDebug -Step     # Step through script
Set-PSDebug -Off      # Turn off debugging

# Verbose output
Write-Verbose "Debug information"
$VerbosePreference = "Continue"

# Debug breakpoints
Set-PSBreakpoint -Script "script.ps1" -Line 10
Set-PSBreakpoint -Script "script.ps1" -Command "Get-Process"

# Logging with different levels
function Write-Log {
    param(
        [string]$Message,
        [ValidateSet("Info", "Warning", "Error", "Debug")]
        [string]$Level = "Info"
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"

    switch ($Level) {
        "Info" { Write-Host $logMessage -ForegroundColor Green }
        "Warning" { Write-Host $logMessage -ForegroundColor Yellow }
        "Error" { Write-Host $logMessage -ForegroundColor Red }
        "Debug" { Write-Debug $logMessage }
    }
}
```

### Batch Script Debugging

```batch
@echo off
REM Batch script debugging

REM Enable command echoing
@echo on

REM Enable delayed expansion for debugging
setlocal enabledelayedexpansion

REM Debug variables
echo DEBUG: Variable value is %variable%

REM Pause for debugging
pause

REM Conditional debugging
if "%DEBUG%"=="1" (
    echo DEBUG: Processing file %1
    echo DEBUG: Current directory: %CD%
)

REM Error level checking
command
if errorlevel 1 (
    echo ERROR: Command failed with error level %errorlevel%
) else (
    echo SUCCESS: Command completed successfully
)
```

---

## Best Practices

### PowerShell Best Practices

```powershell
# Script structure
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$Parameter1,

    [Parameter(Mandatory=$false)]
    [int]$Parameter2 = 10
)

# Error handling
$ErrorActionPreference = "Stop"

# Use approved verbs for functions
function Get-SystemInfo { }
function Set-Configuration { }
function Remove-TempFiles { }

# Use proper parameter validation
param(
    [Parameter(Mandatory=$true)]
    [ValidateNotNullOrEmpty()]
    [string]$Path,

    [Parameter(Mandatory=$false)]
    [ValidateRange(1, 100)]
    [int]$Count = 10
)

# Use pipeline-friendly functions
function Get-FileInfo {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true, ValueFromPipeline=$true)]
        [string]$Path
    )

    process {
        if (Test-Path $Path) {
            Get-Item $Path | Select-Object Name, Length, LastWriteTime
        }
    }
}

# Use proper comment-based help
<#
.SYNOPSIS
    Brief description
.DESCRIPTION
    Detailed description
.PARAMETER ParameterName
    Parameter description
.EXAMPLE
    Example usage
#>
```

### Batch Script Best Practices

```batch
@echo off
REM Use proper header comments
REM Script: example.bat
REM Author: Your Name
REM Date: 2024-01-01
REM Description: Brief description

REM Use setlocal for variable isolation
setlocal enabledelayedexpansion

REM Use meaningful variable names
set "SCRIPT_DIR=%~dp0"
set "LOG_FILE=%SCRIPT_DIR%script.log"

REM Use functions for reusable code
call :log "Script started"

REM Use proper error handling
if not exist "required_file.txt" (
    call :log "ERROR: Required file not found"
    exit /b 1
)

REM Use consistent naming conventions
set "TEMP_DIR=%TEMP%\my_script"
set "BACKUP_DIR=%SCRIPT_DIR%backup"

REM Clean up at the end
call :cleanup
goto :eof

REM Function definitions
:log
echo %date% %time% - %~1 >> "%LOG_FILE%"
goto :eof

:cleanup
if exist "%TEMP_DIR%" rmdir /s /q "%TEMP_DIR%"
goto :eof
```

### Cross-Platform Best Practices

```bash
#!/bin/bash
# Cross-platform script best practices

# Use shebang with env for portability
#!/usr/bin/env bash

# Check for required commands
command -v jq >/dev/null 2>&1 || { echo "jq is required but not installed. Aborting." >&2; exit 1; }

# Use POSIX-compliant syntax when possible
# Good
[ "$var" = "value" ]
# Bad
[[ "$var" == "value" ]]

# Handle different line endings
dos2unix script.sh

# Use absolute paths when possible
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check operating system
case "$(uname -s)" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    CYGWIN*)    MACHINE=Cygwin;;
    MINGW*)     MACHINE=MinGw;;
    *)          MACHINE="UNKNOWN:$(uname -s)"
esac

# Platform-specific commands
if [ "$MACHINE" = "Linux" ]; then
    # Linux-specific code
    :
elif [[ "$MACHINE" == *"MINGW"* ]] || [[ "$MACHINE" == *"CYGWIN"* ]]; then
    # Windows-specific code
    :
fi
```

---

## Quick Reference

### PowerShell Essential Commands

```powershell
# Variables
$var = "value"              # Assignment
"$var"                      # String expansion
${var}                      # Variable expansion

# Conditionals
if (condition) { }          # if statement
switch ($var) { }           # switch statement
for ($i=0; $i-lt5; $i++) { } # for loop
foreach ($item in $list) { } # foreach loop

# Functions
function Name { }           # Function definition
return $value               # Return value

# File operations
Test-Path $path             # Check if path exists
Get-Content $file           # Read file
Set-Content $file $content  # Write file
```

### Batch Essential Commands

```batch
REM Variables
set var=value               REM Assignment
echo %var%                  REM Expansion
set "var=value"             REM Assignment with quotes

REM Conditionals
if condition ( )            REM if statement
for %%i in (list) do ( )    REM for loop
goto label                  REM goto statement

REM Functions
:function_name              REM Function definition
exit /b value               REM Return value

REM File operations
if exist file ( )           REM Check if file exists
type file                   REM Read file
echo text > file            REM Write file
```

### Cross-Platform Commands

```bash
# Variables
var="value"                 # Assignment
echo "$var"                 # Expansion
echo "${var}"               # Variable expansion

# Conditionals
if [ condition ]; then      # if statement
case $var in                # case statement
for item in list; do        # for loop
while [ condition ]; do     # while loop

# Functions
function_name() { }         # Function definition
return $value               # Return value

# File operations
[ -f "$file" ]              # Check if file exists
cat "$file"                 # Read file
echo "text" > "$file"       # Write file
```

This comprehensive shell guide covers all essential aspects of shell scripting on Windows systems, including PowerShell, Command Prompt, and cross-platform solutions. Remember to always test your scripts thoroughly and follow security best practices.
