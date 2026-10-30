"""Check the Windows launcher's actual dry-run output without starting WSL."""
import base64
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(os.name == "nt" and shutil.which("powershell.exe"), "Windows PowerShell native quoting")
class WslLauncherQuotingTests(unittest.TestCase):
    def dry_run(self, arguments):
        script = Path(__file__).resolve().parents[1] / "deploy" / "start_wsl_training.ps1"
        encoded = base64.b64encode(json.dumps(arguments).encode("utf-8")).decode("ascii")
        source = ("$TrainingArgs = [string[]](ConvertFrom-Json ([Text.Encoding]::UTF8.GetString("
                  f"[Convert]::FromBase64String('{encoded}')))); "
                  "& '" + str(script).replace("'", "''") + "' -DryRun -TrainingCommand $TrainingArgs")
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", source],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()

    @staticmethod
    def native_arguments(command_line):
        shell = ctypes.WinDLL("shell32", use_last_error=True)
        shell.CommandLineToArgvW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
        shell.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
        count = ctypes.c_int()
        pointer = shell.CommandLineToArgvW(command_line, ctypes.byref(count))
        if not pointer:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return [pointer[index] for index in range(count.value)]
        finally:
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.LocalFree.argtypes = [ctypes.c_void_p]
            kernel.LocalFree(pointer)

    def test_wsl_switches_are_bare_so_the_first_option_is_recognized(self):
        command_line = self.dry_run(["/usr/bin/python3", "train_5090.py"])
        self.assertIn(" -d Ubuntu -u root -e /usr/bin/python3 ", command_line)
        self.assertNotIn('"-d"', command_line)
        self.assertNotIn('"-e"', command_line)

    def test_spaces_quotes_empty_strings_and_trailing_backslashes_round_trip(self):
        arguments = ["/usr/bin/python3", "plain", "with space", 'quote"inside',
                     "space before trailing \\", "", "$(never_execute)"]
        command_line = self.dry_run(arguments)
        parsed = self.native_arguments(command_line)
        self.assertEqual(parsed[parsed.index("--") + 1:], arguments)


if __name__ == "__main__":
    unittest.main()
