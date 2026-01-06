# LnardoTool - Leonardo AI Batch Generator

Professional Windows desktop application for batch generating product photos using Leonardo AI API.

## Features

- **Workspace Mode**: All work is done in `Documents\LnardoTool` folder
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
   - Create workspace in `Documents\LnardoTool`
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
   - Icon is optional. If you want a custom icon, create `assets/app.ico` locally (not committed to repo).
   - If icon is missing, PyInstaller will use default Python icon.

3. **Build executable:**
   
   **Without icon:**
   ```bash
   pyinstaller --noconsole --onedir --name LnardoTool app.py
   ```
   
   **With icon (if assets/app.ico exists):**
   ```bash
   pyinstaller --noconsole --onedir --name LnardoTool --icon assets/app.ico app.py
   ```

4. **Test the build:**
   - Run `dist\LnardoTool\LnardoTool.exe`
   - Verify workspace is created in `Documents\LnardoTool`

5. **Create installer (optional):**
   - Open `installer/lnardo.iss` in Inno Setup Compiler
   - Build → Compile
   - Installer will be in `installer/` folder

## Workspace Structure

```
Documents\LnardoTool\
├── input\
│   ├── pack\          # Pack reference images: <SKU>_pack.jpg
│   └── piece\          # Piece reference images: <SKU>_piece.jpg
├── output\             # Generated images
│   └── <SKU>__<name>\  # Per-SKU folders
│       ├── <SKU>__pack.png
│       ├── <SKU>__piece.png
│       └── manifest.csv
├── skus.csv            # SKU list (sku, name columns)
└── .env                # API key (created by API Key Wizard, intentionally ignored by git)
```

## Usage

1. **First Run:**
   - App creates workspace in `Documents\LnardoTool`
   - API Key Wizard prompts for Leonardo API key
   - Paste your key and optionally set Model ID
   - **Note:** `.env` file is created locally in `Documents\LnardoTool\.env` by the API Key Wizard
   - `.env` is intentionally ignored by git and must never be committed

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

- **API Key Missing**: Run app and use the API Key Wizard
- **Missing References**: Check `input/pack/` and `input/piece/` folders
- **Generation Fails**: Check log for error messages, verify API key is valid
- **Workspace Not Found**: App creates it automatically on first run

## Requirements

- Windows 10/11
- Leonardo AI API key
- Internet connection

## License

Proprietary - Laurapets internal tool
