# LnardoTool - Leonardo AI Batch Generator

Professional Windows desktop application for batch generating product photos using Leonardo AI API.

## Features

- **Workspace Mode**: All work is done in `workspace/` folder next to the executable
- **Blur-to-Sharp Pipeline**: Automatically enhances blurry reference images
- **Watermark Detection**: Rejects reference images with watermarks/text overlays
- **Studio Photo Mode**: Generates realistic studio product photos
- **Batch Processing**: Process multiple SKUs with continue-on-error
- **Retry Logic**: Automatic retry for transient API errors
- **Per-SKU Tracking**: Individual manifest files for each product

## Installation

### Option 1: Installer (Recommended)

1. Download `LnardoTool-Setup.exe` from releases
2. Run the installer
3. Launch from Desktop shortcut or Start Menu
4. On first run, the app will:
   - Create workspace in `<install_dir>\workspace\`
   - Prompt for Leonardo API key

### Option 2: Build from Source

#### Prerequisites

- Python 3.12+
- PyInstaller
- Inno Setup (for installer)

#### Build Steps

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare icon (optional):**
   - Place `app.ico` in `assets/` folder
   - If missing, PyInstaller will use default icon

3. **Build executable:**
   ```bash
   pyinstaller --noconsole --onedir --name LnardoTool --icon assets/app.ico app.py
   ```

4. **Test the build:**
   - Run `dist\LnardoTool\LnardoTool.exe`
   - Verify workspace is created in `dist\LnardoTool\workspace\`

5. **Create installer (optional):**
   - Open `installer/lnardo.iss` in Inno Setup Compiler
   - Build → Compile
   - Installer will be in `installer/` folder as `LnardoTool-Setup.exe`
   - **IMPORTANT**: Do NOT commit the installer exe to git (it's in .gitignore)

## Workspace Structure

The workspace is created next to the executable (or in the install directory):

```
<app_dir>/workspace/
├── input/
│   ├── pack/          # Pack reference images: <SKU>_pack.jpg
│   └── piece/         # Piece reference images: <SKU>_piece.jpg
├── output/            # Generated images
│   └── <SKU>__<name>/ # Per-SKU folders
│       ├── <SKU>__pack.png
│       ├── <SKU>__piece.png
│       └── manifest.csv
├── skus.csv           # SKU list (sku, name columns)
└── .env               # API key (created by wizard, never committed)
```

**Note**: When running from PyInstaller `dist/`, workspace is at `dist/LnardoTool/workspace/`.
When installed via installer, workspace is at `<install_dir>/workspace/` (typically `%LOCALAPPDATA%\LnardoTool\workspace\`).

## Usage

1. **First Run:**
   - App creates workspace next to executable in `workspace/` folder
   - API Key Wizard prompts for Leonardo API key
   - Paste your key and optionally set Model ID
   - API key is saved to `workspace/.env` (never committed to git)

2. **Prepare References:**
   - Add pack images: `input/pack/<SKU>_pack.jpg`
   - Add piece images: `input/piece/<SKU>_piece.jpg`
   - Update `skus.csv` with SKU and product name

3. **Configure Settings:**
   - Select profile: CHEAP (768×768) or HQ (1024×1024)
   - Adjust init_strength if needed (default 0.90)
   - Enable/disable features (watermark rejection, studio mode, etc.)

4. **Generate:**
   - Click "Validate Batch" to check references
   - Click "Start Generation" to begin
   - Monitor progress in log window
   - Results appear in `output/<SKU>__<name>/`

## Profiles

- **CHEAP**: 768×768, 30 steps, Alchemy ON, init_strength 0.90
- **HQ**: 1024×1024, 30 steps, Alchemy ON, init_strength 0.90

## Troubleshooting

- **API Key Missing**: Run app and use the API Key Wizard (saves to `workspace/.env`)
- **Missing References**: Check `workspace/input/pack/` and `workspace/input/piece/` folders
- **Generation Fails**: Check log for error messages, verify API key is valid
- **Workspace Not Found**: App creates it automatically on first run next to executable
- **Can't Scroll UI**: Use mouse wheel or scrollbar on the right side of the window

## Requirements

- Windows 10/11
- Leonardo AI API key
- Internet connection

## License

Proprietary - Laurapets internal tool
