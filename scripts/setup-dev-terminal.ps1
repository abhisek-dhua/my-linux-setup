# Developer Terminal Setup Script for Windows
# Compatible with PowerShell 5.1 and 7+
# Usage: powershell -ExecutionPolicy Bypass -File C:\Users\Public\setup-dev-terminal.ps1

$ErrorActionPreference = "Stop"

# If running under PS5.1, install/re-launch under PS7
if ($PSVersionTable.PSVersion.Major -lt 7) {
    $ps7 = Get-Command pwsh -ErrorAction SilentlyContinue
    if (!$ps7) {
        Write-Host "`n=== Bootstrapping PowerShell 7 ===" -ForegroundColor Cyan
        Write-Host "Installing PowerShell 7 first..." -ForegroundColor Yellow
        winget install Microsoft.PowerShell --accept-source-agreements --accept-package-agreements
        Write-Host "PowerShell 7 installed!" -ForegroundColor Green
        Write-Host "Please re-run this script manually." -ForegroundColor Yellow
        exit
    }
    Write-Host "`nRe-launching script under PowerShell 7..." -ForegroundColor Yellow
    Start-Process -FilePath $ps7.Source -ArgumentList "-ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Wait
    exit
}

Write-Host "`n=== Developer Terminal Setup ===" -ForegroundColor Cyan

# 1. Install Windows Terminal
Write-Host "`n[1/6] Installing Windows Terminal..." -ForegroundColor Yellow
if (Get-Command wt -ErrorAction SilentlyContinue) {
    Write-Host "  Already installed" -ForegroundColor Green
} else {
    winget install Microsoft.WindowsTerminal --accept-source-agreements --accept-package-agreements
    Write-Host "  Installed!" -ForegroundColor Green
}

# 2. PowerShell 7 already confirmed at this point
Write-Host "`n[2/6] PowerShell 7 - Already installed" -ForegroundColor Green

# 3. Install Oh My Posh
Write-Host "`n[3/6] Installing Oh My Posh..." -ForegroundColor Yellow
if (Get-Command oh-my-posh -ErrorAction SilentlyContinue) {
    Write-Host "  Already installed" -ForegroundColor Green
} else {
    winget install JanDeDobbeleer.OhMyPosh --accept-source-agreements --accept-package-agreements
    Write-Host "  Installed!" -ForegroundColor Green
}

# 4. Install Nerd Font (JetBrainsMono)
Write-Host "`n[4/6] Installing JetBrainsMono Nerd Font..." -ForegroundColor Yellow
$fontPath = "$env:LOCALAPPDATA\Microsoft\Windows\Fonts"
$fontFile = "$fontPath\JetBrainsMonoNerdFont-Regular.ttf"
if (Test-Path $fontFile) {
    Write-Host "  Already installed" -ForegroundColor Green
} else {
    Write-Host "  Downloading font..."
    $fontZip = "$env:TEMP\JetBrainsMono.zip"
    Invoke-WebRequest -Uri "https://github.com/ryanoasis/nerd-fonts/releases/latest/download/JetBrainsMono.zip" -OutFile $fontZip
    Expand-Archive -Path $fontZip -DestinationPath "$env:TEMP\JetBrainsMono" -Force
    if (!(Test-Path $fontPath)) { New-Item -ItemType Directory -Path $fontPath -Force | Out-Null }
    Copy-Item "$env:TEMP\JetBrainsMono\JetBrainsMono Nerd Font*.ttf" -Destination $fontPath -Force
    Write-Host "  Font installed! May require restart to appear" -ForegroundColor Green
}

# 5. Install PowerShell Modules
Write-Host "`n[5/6] Installing PowerShell modules..." -ForegroundColor Yellow
$modules = @("PSReadLine", "posh-git", "Terminal-Icons", "z")
foreach ($mod in $modules) {
    if (Get-Module -ListAvailable -Name $mod) {
        Write-Host "  $mod - Already installed" -ForegroundColor Green
    } else {
        Write-Host "  Installing $mod..."
        Install-Module -Name $mod -Scope CurrentUser -Force -AllowClobber
        Write-Host "  $mod - Installed!" -ForegroundColor Green
    }
}

# 6. Configure PowerShell Profile
Write-Host "`n[6/6] Configuring PowerShell profile..." -ForegroundColor Yellow

$profileDir = Split-Path $PROFILE -Parent
if (!(Test-Path $profileDir)) { New-Item -ItemType Directory -Path $profileDir -Force | Out-Null }

$profileContent = @'
# === Developer Terminal Configuration ===

# Oh My Posh Prompt (v29+ uses --config with built-in theme names)
oh-my-posh init pwsh --config "https://raw.githubusercontent.com/JanDeDobbeleer/oh-my-posh/main/themes/catppuccin_mocha.omp.json" | Invoke-Expression

# Git Integration
Import-Module posh-git

# Terminal Icons
Import-Module Terminal-Icons

# Z Module (directory jumping)
Import-Module z

# PSReadLine Configuration
Set-PSReadLineOption -PredictionSource History
Set-PSReadLineOption -PredictionViewStyle ListView
Set-PSReadLineOption -EditMode Windows
Set-PSReadLineKeyHandler -Key Tab -Function Complete

# Useful Aliases
Set-Alias -Name ll -Value Get-ChildItem
function up { Set-Location .. }
function upup { Set-Location ../.. }

# Quick Navigation
function cdp { Set-Location $HOME\Desktop }
function cdd { Set-Location $HOME\Downloads }
function cddocs { Set-Location $HOME\Documents }

# Git Shortcuts
function gs { git status }
function ga { git add . }
function gc { param([Parameter(Mandatory)][string]$m); git commit -m $m }
function gp { git push }
function gl { git log --oneline -10 }

Write-Host "Ready!" -ForegroundColor Green

# === Utility Functions ===

# Change Oh My Posh theme
# Usage: Set-Theme "tokyonight_storm" or Set-Theme "catppuccin_mocha"
function Set-Theme {
    param([Parameter(Mandatory)][string]$Name)
    $themeUrl = "https://raw.githubusercontent.com/JanDeDobbeleer/oh-my-posh/main/themes/$Name.omp.json"
    $test = Invoke-WebRequest -Uri $themeUrl -Method Head -ErrorAction SilentlyContinue
    if ($test.StatusCode -ne 200) {
        Write-Host "Theme '$Name' not found. Browse themes at: https://ohmyposh.dev/docs/themes" -ForegroundColor Red
        return
    }
    $profilePath = $PROFILE
    $content = Get-Content $profilePath -Raw
    $pattern = 'oh-my-posh init pwsh --config ".*?"'
    $replacement = "oh-my-posh init pwsh --config `"$themeUrl`""
    $content = $content -replace $pattern, $replacement
    Set-Content -Path $profilePath -Value $content -Force
    Write-Host "Theme changed to '$Name'. Restart terminal to apply." -ForegroundColor Green
}

# List popular themes
function Get-Themes {
    $themes = @(
        "catppuccin_mocha", "catppuccin_latte", "tokyonight_storm", "tokyonight_night",
        "dracula", "powerlevel10k_rainbow", "powerlevel10k_classic", "agnoster",
        "agnosterplus", "avit", "blueish", "bubbles", "bubblesline", "cert",
        "chips", "craver", "darkblood", "devious-diamonds", "emodipt",
        "froczh", "free-ukraine", "gruvbox", "half-life", "huvix",
        "iterm2", "jandedobbeleer", "jblab_2021", "jonnychipz", "json",
        "jtracey93", "jv_sitecorian", "kushal", "lambda", "marcduiker",
        "material", "microverse-power", "monokai", "mt", "negligee",
        "nu4a", "onehalf.minimal", "paradox", "patriksvensson", "poshmon",
        "pure", "quick-term", "remk", "robbyrussell", "rudolfs-dark",
        "rudolfs-light", "sim-web", "slim", "slimfat", "smoothie",
        "sonicboom_dark", "sonicboom_light", "sorin", "space", "spaceship",
        "star", "stelbent", "stelbent.minimal", "takuya", "the-unnamed",
        "tiare", "tokyo", "tonybaloney", "velvet", "wholespace", "wopian",
        "ys", "zash"
    )
    Write-Host "`nAvailable themes (use Set-Theme <name>):" -ForegroundColor Cyan
    $themes | ForEach-Object { Write-Host "  $_" -ForegroundColor White }
    Write-Host "`nBrowse all: https://ohmyposh.dev/docs/themes" -ForegroundColor Yellow
}

# Open Windows Terminal settings
function Edit-Terminal {
    wt settings
}

# Open PowerShell profile
function Edit-Profile {
    code $PROFILE
}

# Reload profile
function Reload {
    . $PROFILE
}

# Show this help
function TerminalHelp {
    Write-Host "`n=== Terminal Commands ===" -ForegroundColor Cyan
    Write-Host "  Set-Theme <name>   - Change prompt theme (e.g. Set-Theme dracula)" -ForegroundColor White
    Write-Host "  Get-Themes         - List available themes" -ForegroundColor White
    Write-Host "  Edit-Terminal      - Open Windows Terminal settings" -ForegroundColor White
    Write-Host "  Edit-Profile       - Open PowerShell profile in VS Code" -ForegroundColor White
    Write-Host "  Reload             - Reload profile without restarting" -ForegroundColor White
    Write-Host "  TerminalHelp       - Show this help" -ForegroundColor White
    Write-Host "=====================`n" -ForegroundColor Cyan
}
'@

if (Test-Path $PROFILE) {
    Write-Host "  Profile already exists, backing up to $PROFILE.bak" -ForegroundColor Yellow
    Copy-Item $PROFILE "$PROFILE.bak" -Force
}

Set-Content -Path $PROFILE -Value $profileContent -Force
Write-Host "  Profile configured!" -ForegroundColor Green

# Configure Windows Terminal settings
Write-Host "`nConfiguring Windows Terminal defaults..." -ForegroundColor Yellow

$wtSettingsPath = "$env:LOCALAPPDATA\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json"
if (Test-Path $wtSettingsPath) {
    $wtConfig = Get-Content $wtSettingsPath | ConvertFrom-Json

    # Set defaults
    $wtConfig.defaults | Add-Member -NotePropertyName "font" -NotePropertyValue @{ face = "JetBrainsMono Nerd Font"; size = 11 } -Force
    $wtConfig.defaults | Add-Member -NotePropertyName "useAcrylic" -NotePropertyValue $true -Force
    $wtConfig.defaults | Add-Member -NotePropertyName "acrylicOpacity" -NotePropertyValue 0.85 -Force
    $wtConfig.defaults | Add-Member -NotePropertyName "colorScheme" -NotePropertyValue "Catppuccin Mocha" -Force
    $wtConfig.defaults | Add-Member -NotePropertyName "padding" -NotePropertyValue "12" -Force

    # Add Catppuccin Mocha theme if not present
    $catppuccinMocha = @{
        name = "Catppuccin Mocha"
        cursorColor = "#F5E0DC"
        selectionBackground = "#585B70"
        background = "#1E1E2E"
        foreground = "#CDD6F4"
        black = "#45475A"
        blue = "#89B4FA"
        cyan = "#94E2D5"
        green = "#A6E3A1"
        purple = "#F5C2E7"
        red = "#F38BA8"
        white = "#BAC2DE"
        yellow = "#F9E2AF"
        brightBlack = "#585B70"
        brightBlue = "#89B4FA"
        brightCyan = "#94E2D5"
        brightGreen = "#A6E3A1"
        brightPurple = "#F5C2E7"
        brightRed = "#F38BA8"
        brightWhite = "#A6ADC8"
        brightYellow = "#F9E2AF"
    }

    $exists = $wtConfig.schemes | Where-Object { $_.name -eq "Catppuccin Mocha" }
    if (!$exists) {
        $wtConfig.schemes += $catppuccinMocha
    }

    $wtConfig | ConvertTo-Json -Depth 10 | Set-Content $wtSettingsPath -Force
    Write-Host "  Windows Terminal configured!" -ForegroundColor Green
} else {
    Write-Host "  Windows Terminal settings not found. Open WT once to generate, then run again." -ForegroundColor Yellow
}

Write-Host "`n=== Setup Complete! ===" -ForegroundColor Cyan
Write-Host "Restart your terminal to apply all changes." -ForegroundColor White
Write-Host "Theme: Catppuccin Mocha | Font: JetBrainsMono Nerd Font" -ForegroundColor White
