# Ship-Ready Checklist - Final Release

## ✅ All Critical Fixes Applied

### 1. ✅ Critical Bug Fixed
- `self.lockable_widgets` initialized BEFORE `_build_ui()`
- `self.run_log_fp` initialized BEFORE `_build_ui()`
- App no longer crashes on startup

### 2. ✅ API Key Wizard Implemented
- `ensure_api_key_ui()` method added
- Uses `simpledialog.askstring()` for clean modal input
- Called in `on_start()` and `on_test_api()` before worker thread
- Saves to workspace `.env` automatically
- No more unprofessional error messages

### 3. ✅ Thread-Safe UI Updates
- Added `ui()`, `set_status()`, `set_progress()` helper methods
- All `status_var.set()` and `progress_var.set()` calls in `_run_batch()` replaced
- All `messagebox` calls in worker thread wrapped with `self.ui(lambda: ...)`
- Production-grade stability achieved

## Build & Install Instructions

### Step 1: Build Executable

```bash
# Install dependencies
pip install -r requirements.txt

# Build with PyInstaller
pyinstaller --noconsole --onedir --name LnardoTool --icon assets/app.ico app.py
```

**Expected output:**
- `dist/LnardoTool/` folder with executable

### Step 2: Test Build Locally

1. Run `dist\LnardoTool\LnardoTool.exe`
2. Verify:
   - ✅ App launches without crash
   - ✅ Workspace created in `dist\LnardoTool\workspace\`
   - ✅ API Key Wizard appears if .env missing
   - ✅ Profile switching works (CHEAP/HQ)
   - ✅ UI scrolls properly (mouse wheel works, scrollbar visible)
   - ✅ UI locks during generation
   - ✅ Generation works end-to-end

### Step 3: Create Installer

1. **Install Inno Setup:**
   - Download from https://jrsoftware.org/isinfo.php
   - Install Inno Setup Compiler

2. **Compile installer:**
   - Open `installer/lnardo.iss` in Inno Setup
   - Build → Compile
   - Output: `installer/LnardoTool-Setup.exe`
   - **IMPORTANT**: Do NOT commit `installer/LnardoTool-Setup.exe` to git (it's in .gitignore)

### Step 4: Copy to USB

1. Copy `installer/LnardoTool-Setup.exe` to USB drive
2. Optionally include:
   - `README.md` (usage instructions)
   - `BUILD_CHECKLIST.md` (for reference)

### Step 5: Install on Simona's Notebook

1. Insert USB drive
2. Run `LnardoTool-Setup.exe`
3. Follow installer (defaults are fine)
4. Launch from Desktop shortcut
5. On first run:
   - Workspace auto-created in `<install_dir>\workspace\` (typically `%LOCALAPPDATA%\LnardoTool\workspace\`)
   - API Key Wizard prompts for key
   - Paste Leonardo API key
   - App ready to use

## Final Verification

After installation on Simona's notebook:

- [ ] App launches without console window
- [ ] Workspace created in `<install_dir>\workspace\` (not Documents)
- [ ] API Key Wizard works on first run
- [ ] UI scrolls properly (mouse wheel works, scrollbar visible)
- [ ] Profile switching updates UI correctly
- [ ] UI locks during generation
- [ ] Generation completes successfully
- [ ] Per-SKU manifest created with correct schema
- [ ] Run log file created in output folder
- [ ] Continue-on-error works (test with missing ref)

## File Structure on Simona's Notebook

```
C:\Users\Simona\AppData\Local\LnardoTool\
├── LnardoTool.exe
├── [app files from installer]
└── workspace\
    ├── input\
    │   ├── pack\
    │   └── piece\
    ├── output\
    │   └── run_YYYYMMDD_HHMMSS.log
    ├── skus.csv
    └── .env
```

## Notes

- App is fully self-contained after installation
- All work happens in `<install_dir>\workspace\` (not Program Files, not Documents)
- API key stored securely in `workspace/.env` (never committed to git)
- No console window (professional appearance)
- Entire UI is scrollable with mouse wheel support (Windows)
- Thread-safe UI updates prevent random crashes
- Continue-on-error ensures batch completes even with failures

**Status: ✅ SHIP-READY**
