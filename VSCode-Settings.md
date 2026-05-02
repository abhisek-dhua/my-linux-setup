# VS Code Settings Configuration

This is my personal VS Code settings configuration optimized for web development with Angular, TypeScript, and modern JavaScript workflows.

## Full Settings JSON

```json
{
  // editor customization
  "git.autofetch": true,
  "editor.wordWrap": "on",
  "git.confirmSync": false,
  "chat.agent.enabled": false,
  "editor.formatOnSave": true,
  "chat.editor.enabled": false,
  "editor.fontLigatures": true,
  "editor.linkedEditing": true,
  "workbench.colorTheme": "Dark+",
  "database-client.autoSync": true,
  "gitlens.codeLens.enabled": false,
  "editor.cursorBlinking": "expand",
  "chat.commandCenter.enabled": false,
  "css.lint.unknownAtRules": "ignore",
  "scss.lint.unknownAtRules": "ignore",
  "editor.inlineSuggest.enabled": false,
  "editor.guides.bracketPairs": "active",
  "cSpell.userWords": ["Abhisek", "Dhua"],
  "js/ts.updateImportsOnPaste.enabled": true,
  "workbench.iconTheme": "material-icon-theme",
  "liveServer.settings.donotShowInfoMsg": true,
  "editor.bracketPairColorization.enabled": true,
  "terminal.integrated.defaultProfile.linux": "zsh",
  "terminal.integrated.sendKeybindingsToShell": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "js/ts.updateImportsOnFileMove.enabled": "always",
  "workbench.secondarySideBar.defaultVisibility": "hidden",
  "terminal.integrated.defaultProfile.windows": "PowerShell",
  "editor.bracketPairColorization.independentColorPoolPerBracketType": true,
  "editor.fontFamily": "'FiraCode Nerd Font', Consolas, 'Courier New', monospace",
  // prettier customization
  "prettier.semi": true,
  "prettier.tabWidth": 2,
  "prettier.useTabs": false,
  "prettier.endOfLine": "lf",
  "prettier.printWidth": 100,
  "prettier.singleQuote": true,
  "prettier.trailingComma": "all",
  "prettier.bracketSpacing": true,
  "prettier.arrowParens": "always",
  "prettier.bracketSameLine": true,
  "prettier.quoteProps": "as-needed",
  "prettier.embeddedLanguageFormatting": "auto",
  "prettier.htmlWhitespaceSensitivity": "ignore",
  // vscode workbench customization
  "workbench.colorCustomizations": {
    "statusBar.background": "#2d2d2d",
    "statusBar.noFolderBackground": "#2d2d2d",
    "statusBar.debuggingBackground": "#c74e39",
    "statusBar.foreground": "#cccccc"
  },
  "editor.tokenColorCustomizations": {
    "textMateRules": [
      {
        "scope": [
          // the following elements will be in italic
          "comment",
          "storage.modifier", // static keyword
          "storage.type.php", // typehints in methods keyword
          "keyword.other.new.php", // new
          "entity.other.attribute-name", // html attributes
          "fenced_code.block.language.markdown", // markdown language modifier
          "keyword", //import, export, return…
          "storage.type", //class keyword
          "storage.modifier", //static keyword
          "keyword.control",
          "constant.language",
          "entity.name.method",
          "entity.name.function",
          "entity.other.attribute-name",
          "keyword.control.import.ts",
          "keyword.control.import.tsx",
          "keyword.control.import.js",
          "keyword.control.flow.js",
          "keyword.control.from.js",
          "keyword.control.from.ts",
          "keyword.control.from.tsx"
        ],
        "settings": {
          "fontStyle": "italic"
          // "foreground": "#82AAFF"
        }
      },
      {
        "scope": [
          // the following elements will be displayed in bold
          "entity.name.type.class" // class names
        ],
        "settings": {
          "fontStyle": ""
        }
      },
      {
        "scope": [
          // the following elements will be displayed in bold and italic
          "entity.name.section.markdown" // markdown headlines
        ],
        "settings": {
          "fontStyle": "italic"
        }
      },
      {
        "scope": [
          // the following elements will be excluded from italics
          //   (VSCode has some defaults for italics)
          "invalid",
          "comment.block",
          "keyword.operator",
          "constant.numeric.css",
          "constant.numeric.json",
          "keyword.other.unit.px.css",
          "constant.numeric.decimal.js",
          "entity.other.attribute-name.class.css"
        ],
        "settings": {
          "fontStyle": ""
        }
      }
    ]
  },
  // trusted json schemas
  "json.schemaDownload.trustedDomains": {
    "https://turbo.build": true,
    "https://json-schema.org/": true,
    "https://www.schemastore.org/": true,
    "https://json.schemastore.org/": true,
    "https://raw.githubusercontent.com/": true,
    "https://schemastore.azurewebsites.net/": true
  }
}
```

---

## Required Extensions

To use these settings properly, install the following VS Code extensions:

1. **Prettier - Code Formatter** - `esbenp.prettier-vscode`
2. **Material Icon Theme** - `PKief.material-icon-theme`
3. **GitLens** - `eamodio.gitlens`
4. **Database Client** - `cweijan.vscode-mysql-client2`
5. **Live Server** - `ritwickdey.LiveServer`
6. **Code Spell Checker** - `streetsidesoftware.code-spell-checker`

---

## Key Features

### Editor

- Format on save enabled
- FiraCode Nerd Font with font ligatures
- Bracket pair colorization with active guides
- Linked editing for matching tags
- Word wrap enabled
- Expand cursor blinking style
- Inline suggestions disabled

### Terminal

- Zsh as default Linux shell
- PowerShell as default Windows shell
- Keybindings sent directly to shell

### Git

- Auto fetch enabled
- Sync confirmation disabled
- GitLens code lens disabled

### Prettier

- 2 space indentation
- Single quotes
- Semicolons enabled
- Trailing commas on all
- Bracket spacing enabled
- Print width set to 100 characters
- LF line endings

### Theme & Appearance

- Dark+ color theme
- Material Icon Theme
- Custom status bar styling
- Custom syntax highlighting with italic keywords
- Hidden secondary sidebar by default

---

**Last Updated:** April 9, 2026
