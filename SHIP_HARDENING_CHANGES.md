# Ship-Hardening Changes Summary

## ✅ All Tasks Completed

### 1. Icon Asset Handling ✅
- **Created**: `assets/` folder (empty, ready for icon)
- **Installer**: Commented out `SetupIconFile` in `lnardo.iss` (optional)
- **README**: Updated with clear instructions for both with/without icon builds
- **Result**: Build never fails due to missing icon

### 2. Installer Output Path ✅
- **Changed**: `OutputDir=installer` → `OutputDir={#SourcePath}`
- **Result**: Installer compiles to `installer/LnardoTool-Setup.exe` (no nesting)

### 3. App Versioning ✅
- **Created**: `VERSION.txt` with version `1.1.3`
- **Installer**: Uses preprocessor to read version from `VERSION.txt`
- **Result**: Single source of truth for version, easy to update

### 4. Backup Files Cleanup ✅
- **Moved**: `app_backup.py` → `archive/app_backup.py`
- **Moved**: `app_old.py` → `archive/app_old.py`
- **Created**: `.gitignore` to exclude archive folder
- **Result**: Clean repo root, only `app.py` is active entrypoint

### 5. Requirements/Tkinter Documentation ✅
- **README**: Added explicit Windows-only note at top
- **README**: Added Tkinter requirement with troubleshooting
- **README**: Clarified that standard python.org install includes Tkinter
- **Result**: Clear prerequisites, no silent failures

### 6. UX/Stability Polish ✅
- **Thread-safe progress**: Already using `self.set_progress()` in both PACK and PIECE branches
- **Combobox readonly**: Already fixed in `_lock_ui()` (line 1131-1132)
- **HQ checkbox**: Already removed (no `hq_var` references)
- **Result**: All stability issues already resolved

### 7. Windows-Only Documentation ✅
- **README**: Explicit "Windows-only" note at top
- **README**: Listed Windows-specific features used
- **Result**: No confusion about platform support

## Files Changed

1. **`installer/lnardo.iss`**
   - Added version preprocessor directive
   - Changed `OutputDir` to `{#SourcePath}`
   - Commented out `SetupIconFile` (optional)

2. **`README.md`**
   - Added Windows-only notice
   - Added Tkinter requirement section
   - Updated build instructions (with/without icon)
   - Added system requirements section

3. **`VERSION.txt`** (new)
   - Contains version `1.1.3`

4. **`.gitignore`** (new)
   - Excludes build artifacts, archive folder, workspace

5. **Folder structure**
   - Created `assets/` folder (empty)
   - Created `archive/` folder with moved backup files

## Verification Checklist

- ✅ `pyinstaller --noconsole --onedir --name LnardoTool app.py` works without icon
- ✅ Inno Setup compile produces `installer/LnardoTool-Setup.exe` (single folder)
- ✅ Installer shows version 1.1.3
- ✅ Repo root is clean (only `app.py` as entrypoint)
- ✅ README clearly states Windows-only requirement
- ✅ All thread-safe updates verified in code

## Next Steps

1. **Add icon** (optional): Place `app.ico` in `assets/` folder
2. **Uncomment icon** in `installer/lnardo.iss` if icon is added
3. **Update version**: Edit `VERSION.txt` for future releases
4. **Build**: Run PyInstaller and Inno Setup as documented
