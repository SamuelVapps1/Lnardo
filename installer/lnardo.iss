; Inno Setup Script for LnardoTool
; Compile with Inno Setup Compiler (https://jrsoftware.org/isinfo.php)

#define SourcePath ".."
#define MyAppVersion "1.1.2"
#define MyIconPath SourcePath + "/assets/app.ico"


[Setup]
AppName=LnardoTool
AppVersion={#MyAppVersion}
DefaultDirName={localappdata}\LnardoTool
DefaultGroupName=LnardoTool
OutputDir={#SourcePath}
OutputBaseFilename=LnardoTool-Setup
Compression=lzma
SolidCompression=yes
#if FileExists(MyIconPath)
SetupIconFile={#MyIconPath}
#endif
UninstallDisplayIcon={app}\LnardoTool.exe
PrivilegesRequired=lowest

[Files]
Source: "..\dist\LnardoTool\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\LnardoTool"; Filename: "{app}\LnardoTool.exe"; IconFilename: "{app}\LnardoTool.exe"
Name: "{commondesktop}\LnardoTool"; Filename: "{app}\LnardoTool.exe"; IconFilename: "{app}\LnardoTool.exe"
Name: "{group}\Uninstall LnardoTool"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\LnardoTool.exe"; Description: "Launch LnardoTool"; Flags: nowait postinstall skipifsilent
