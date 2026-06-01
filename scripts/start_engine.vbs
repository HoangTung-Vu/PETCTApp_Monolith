Dim fso, wsh, dir
Set fso = CreateObject("Scripting.FileSystemObject")
Set wsh = CreateObject("WScript.Shell")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
wsh.Run """" & dir & "\start_engine.bat""", 0, False
