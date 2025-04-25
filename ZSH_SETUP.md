# Oh my zsh.

## Install with curl
```
sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
```
## Change Theme 

`ZSH_THEME="agnoster"`

## Enabling Plugins (zsh-autosuggestions, zsh-syntax-highlighting & fast-syntax-highlighting)
 - Download zsh-autosuggestions by
 
 ```git clone https://github.com/zsh-users/zsh-autosuggestions.git $ZSH_CUSTOM/plugins/zsh-autosuggestions```
 
 - Download zsh-syntax-highlighting by
 
 ```git clone https://github.com/zsh-users/zsh-syntax-highlighting.git $ZSH_CUSTOM/plugins/zsh-syntax-highlighting```

 - Download fast-syntax-highlighting by
 
 ```git clone https://github.com/zdharma-continuum/fast-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/fast-syntax-highlighting```

 - `nano ~/.zshrc` find `plugins=(git)`
 
 - Append `zsh-autosuggestions, zsh-syntax-highlighting & fast-syntax-highlighting` to  `plugins()` like this 
 
 ```plugins=(git zsh-autosuggestions zsh-syntax-highlighting fast-syntax-highlighting)```

```
# User configuration
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This load>

# Load Angular CLI autocompletion.
source <(ng completion script)
```
 
 - Reload terminal `source ~/.zshrc`

### Ref
 - [oh-my-zsh](https://github.com/ohmyzsh/ohmyzsh)
 - [oh-my-zsh](https://github.com/robbyrussell/oh-my-zsh)
 - [zsh-autosuggestions](https://github.com/zsh-users/zsh-autosuggestions)
 - [zsh-syntax-highlighting](https://github.com/zsh-users/zsh-syntax-highlighting)
