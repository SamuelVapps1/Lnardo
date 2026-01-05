# Build Checklist for LnardoTool

## Pre-Build Steps

1. **Verify dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare icon (optional):**
   - Place `app.ico` in `assets/` folder
   - If missing, PyInstaller will use default Python icon

## Build Executable

```bash
pyinstaller --noconsole --onedir --name LnardoTool --icon assets/app.ico app.py
```

**Expected output:**
- `dist/LnardoTool/` folder with executable and dependencies
- `build/` folder (can be deleted after build)

## Test Build

1. Run `dist\LnardoTool\LnardoTool.exe`
2. Verify:
   - ✅ Workspace created in `Documents\LnardoTool`
   - ✅ API Key Wizard appears if .env missing
   - ✅ Profile dropdown changes settings (CHEAP/HQ)
   - ✅ UI locks during generation
   - ✅ Generation works and creates per-SKU manifest

## Create Installer

1. **Install Inno Setup:**
   - Download from https://jrsoftware.org/isinfo.php
   - Install Inno Setup Compiler

2. **Compile installer:**
   - Open `installer/lnardo.iss` in Inno Setup
   - Build → Compile
   - Installer will be in `installer/` folder as `LnardoTool-Setup.exe`

3. **Test installer:**
   - Run installer on clean system (or VM)
   - Verify desktop shortcut created
   - Verify app launches and creates workspace

## Sanity Test After Build

- [ ] Launch app, confirm workspace created in Documents
- [ ] Confirm preset changes UI values (CHEAP → 768×768, HQ → 1024×1024)
- [ ] Confirm generation works and per-SKU manifest appears in SKU folder
- [ ] Confirm API key wizard works on first run
- [ ] Confirm UI locks during generation
- [ ] Confirm continue-on-error works (test with invalid SKU)

## File Structure After Build

```
dist/LnardoTool/
├── LnardoTool.exe
├── _internal/
│   ├── app.py (compiled)
│   ├── [dependencies]
│   └── ...
└── [other PyInstaller files]
```

## Notes

- App runs without console window (--noconsole flag)
- Workspace is always in `Documents\LnardoTool` regardless of install path
- API key is stored in workspace `.env` file
- Each SKU gets its own folder with manifest.csv
