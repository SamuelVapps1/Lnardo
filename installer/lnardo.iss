; Inno Setup script for LnardoTool
; Compile with: Inno Setup Compiler -> Build -> Compile

#define AppName "LnardoTool"
#define AppPublisher "Laurapets"
#define AppURL "https://github.com/SamuelVapps1/Lnardo"
#define SourcePath ".."

; Read version from VERSION.txt (simple approach, avoid ISPP pitfalls)
#define VersionFile SourcePath + "/VERSION.txt"
#if FileExists(VersionFile)
  #define VersionFileHandle FileOpen(VersionFile)
  #if VersionFileHandle != 0
    #define AppVersionStr FileRead(VersionFileHandle)
    #expr FileClose(VersionFileHandle)
    #define AppVersion Trim(AppVersionStr)
  #else
    #define AppVersion "1.0.0"
  #endif
#else
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
DefaultDirName={localappdata}\{#AppName}
DefaultGroupName={#AppName}
OutputDir=.
OutputBaseFilename=LnardoTool-Setup
Compression=lzma
SolidCompression=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

; Conditional icon (only if exists)
#if FileExists(SourcePath + "/assets/app.ico")
  SetupIconFile={#SourcePath}/assets/app.ico
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 0,6.1

[Files]
Source: "{#SourcePath}/dist/LnardoTool/*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\LnardoTool.exe"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\LnardoTool.exe"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#AppName}"; Filename: "{app}\LnardoTool.exe"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\LnardoTool.exe"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
