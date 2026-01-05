; Inno Setup Script for LnardoTool
; Compile with Inno Setup Compiler (https://jrsoftware.org/isinfo.php)

[Setup]
AppName=LnardoTool
AppVersion=1.0
DefaultDirName={pf}\LnardoTool
DefaultGroupName=LnardoTool
OutputDir=installer
OutputBaseFilename=LnardoTool-Setup
Compression=lzma
SolidCompression=yes
SetupIconFile=..\assets\app.ico
UninstallDisplayIcon={app}\LnardoTool.exe

[Files]
Source: "..\dist\LnardoTool\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\LnardoTool"; Filename: "{app}\LnardoTool.exe"; IconFilename: "{app}\LnardoTool.exe"
Name: "{commondesktop}\LnardoTool"; Filename: "{app}\LnardoTool.exe"; IconFilename: "{app}\LnardoTool.exe"
Name: "{group}\Uninstall LnardoTool"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\LnardoTool.exe"; Description: "Launch LnardoTool"; Flags: nowait postinstall skipifsilent
