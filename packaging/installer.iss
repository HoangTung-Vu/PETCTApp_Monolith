; Inno Setup 6 script for the PET/CT desktop app.
; Run by packaging/build_windows.ps1 after PyInstaller has produced dist\PETCTApp:
;   iscc /DAppVersion=1.0.0 packaging\installer.iss
; Relative paths below are relative to this file.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
; No "/" here: the name is also used for shortcut file names.
#define AppName "PET-CT App"
#define AppExeName "PETCTApp.exe"

[Setup]
; Never change AppId: upgrades and the uninstaller find the install through it.
AppId={{7C539C26-136A-4D79-858E-089C6BBEDBE1}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=HoangTung-Vu
DefaultDirName={autopf}\PETCTApp
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Per-user install by default (no admin needed on hospital PCs); the user can
; still pick "install for all users" if they have admin rights.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist\installer
OutputBaseFilename=PETCTApp-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExeName}
#if FileExists(SourcePath + "assets\petct.ico")
SetupIconFile=assets\petct.ico
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
FinishedLabel=Setup has finished installing [name] on your computer.%n%nYour data folder (the session database petct.db) is never removed by uninstalling or upgrading.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; Clear the previous version's libraries so an upgrade doesn't mix old and new DLLs.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "..\dist\PETCTApp\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

; Deliberately no [UninstallDelete]: the data folder and the saved settings
; (HKCU\Software\PETCTApp) hold patient sessions and must survive uninstall.
