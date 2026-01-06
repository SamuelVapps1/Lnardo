# Final Build & Release Checklist

## Build Executable

```bash
pyinstaller --noconsole --onedir --name LnardoTool --icon assets/app.ico app.py
```

**Expected output:**
- `dist/LnardoTool/` folder with `LnardoTool.exe` and dependencies

## Compile Installer

1. Open `installer/lnardo.iss` in Inno Setup Compiler
2. Build → Compile
3. Installer will be in `installer/` folder as `LnardoTool-Setup.exe`
4. **IMPORTANT**: Do NOT commit `installer/LnardoTool-Setup.exe` to git (it's in .gitignore)

## Quick Sanity Tests

### Test 1: First Run
- [ ] Launch `dist\LnardoTool\LnardoTool.exe`
- [ ] Verify workspace created in `dist\LnardoTool\workspace\`
- [ ] Verify API Key Wizard appears if .env missing
- [ ] Enter API key and verify it saves to `workspace\.env`
- [ ] Verify UI is scrollable (mouse wheel works, scrollbar visible on right)

### Test 2: Profile Switching
- [ ] Select "CHEAP" profile from dropdown
- [ ] Verify: width=768, height=768, steps=30, alchemy=ON, strength=0.90
- [ ] Select "HQ" profile from dropdown
- [ ] Verify: width=1024, height=1024, steps=30, alchemy=ON, strength=0.90
- [ ] Click "Preset: CHEAP" button
- [ ] Verify settings update correctly

### Test 3: UI Locking
- [ ] Click "Start Generation" (with valid setup)
- [ ] Verify: Start button disabled, Stop button enabled
- [ ] Verify: All settings controls disabled (width, height, steps, checkboxes, etc.)
- [ ] Click "Stop"
- [ ] Verify: UI unlocks, Start enabled, Stop disabled

### Test 4: Generation & Manifest
- [ ] Add test SKU to skus.csv
- [ ] Add pack and piece reference images
- [ ] Run generation
- [ ] Verify: Output folder created as `{SKU}__{name}`
- [ ] Verify: `manifest.csv` exists in SKU folder
- [ ] Verify: Manifest has 8 columns: sku,name,variant,generation_id,api_credit_cost,file_path,status,error
- [ ] Verify: Successful entries show status="OK"
- [ ] Verify: Run log file created in output folder

### Test 5: Continue-on-Error
- [ ] Add SKU with missing pack ref
- [ ] Run generation
- [ ] Verify: Missing ref logged as SKIPPED in manifest
- [ ] Verify: Generation continues to next SKU
- [ ] Verify: Batch completes without crashing

### Test 6: Installer
- [ ] Run installer on clean system (or VM)
- [ ] Verify: Desktop shortcut created
- [ ] Verify: Start Menu shortcut created
- [ ] Verify: App launches from shortcut
- [ ] Verify: Workspace created in Documents (not Program Files)

## Release Notes

All blockers fixed:
- ✅ Start/Stop button states and references
- ✅ UI locking with whitelist approach
- ✅ Per-SKU manifest schema (8 columns)
- ✅ Continue-on-error for pack/piece generation
- ✅ Run-level log file
- ✅ Duplicate apply_profile call removed
- ✅ Installer script verified (.iss extension)
