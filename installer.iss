; Inno Setup script for Stream Voice Focus.
; Builds StreamVoiceFocus-Setup-<version>.exe from the PyInstaller output in dist\StreamVoiceFocus.
; Build: build_installer.bat   (needs Inno Setup 6: winget install JRSoftware.InnoSetup)

#define MyAppName "Stream Voice Focus"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Roy Mutwiri"
#define MyAppURL "https://github.com/Roy-Mutwiri/Sound-Twitch-Filter"
#define MyAppExeName "StreamVoiceFocus.exe"

[Setup]
AppId={{7E1D4C2A-6B0F-4D3E-9B5A-3F2C1A8E5D11}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=LICENSE
OutputDir=installer_out
OutputBaseFilename=StreamVoiceFocus-Setup-{#MyAppVersion}
SetupIconFile=assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "dist\StreamVoiceFocus\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Get VB-Cable (required, free)"; Filename: "https://vb-audio.com/Cable/"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
Filename: "https://vb-audio.com/Cable/"; Description: "Open the VB-Cable download page (needed once; install as administrator, then reboot)"; Flags: shellexec postinstall skipifsilent unchecked

[Messages]
FinishedLabel=Setup has finished installing [name] on your computer.%n%nBefore first use:%n1. Install VB-Cable (free) from vb-audio.com/Cable if you have not yet, then reboot.%n2. In Windows Settings > System > Sound > Volume mixer, set your browser's Output to "CABLE Input".%n3. Start [name], choose your headphones as Output, press Start.%n%nThe first Start downloads the speech model (about 30 MB) and then works offline.
