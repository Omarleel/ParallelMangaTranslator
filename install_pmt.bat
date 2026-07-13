@echo off
setlocal
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 install_pmt.py %*
) else (
  python install_pmt.py %*
)
endlocal
