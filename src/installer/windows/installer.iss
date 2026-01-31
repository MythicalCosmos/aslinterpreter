[Setup]
AppName=MyApp
AppVersion=1.0
DefaultDirName={pf}\MyApp
OutputBaseFilename=MyAppInstaller

[Files]
Source: "..\..\frontend-desktop\dist\main.exe"; DestDir: "{app}"

[Icons]
Name: "{group}\MyApp"; Filename: "{app}\main.exe"
