@echo off
REM =============================================================================
REM MLB Win Probability — Local Setup & Start Script (Windows)
REM
REM Usage:
REM   start.bat              Full setup + start
REM   start.bat --stop       Stop running containers
REM   start.bat --status     Check container status
REM   start.bat --rebuild    Force rebuild the Docker image
REM   start.bat --model xgboost  Start with a specific model type
REM
REM Prerequisites:
REM   - Docker Desktop for Windows
REM     https://docs.docker.com/desktop/install/windows-install/
REM =============================================================================

setlocal enabledelayedexpansion

set "MODEL=stacked"
set "PORT=30087"
set "REBUILD=false"
set "ACTION=start"

REM Parse arguments
:parse_args
if "%~1"=="" goto :end_parse
if /i "%~1"=="--stop"      ( set "ACTION=stop"    & shift & goto :parse_args )
if /i "%~1"=="--status"    ( set "ACTION=status"   & shift & goto :parse_args )
if /i "%~1"=="--rebuild"   ( set "REBUILD=true"    & shift & goto :parse_args )
if /i "%~1"=="--model"     ( set "MODEL=%~2"       & shift & shift & goto :parse_args )
if /i "%~1"=="--port"      ( set "PORT=%~2"        & shift & shift & goto :parse_args )
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="-h" goto :show_help
echo [ERROR] Unknown option: %~1
exit /b 1
:end_parse

REM ── Banner ───────────────────────────────────────────────────────────────
echo.
echo   MLB Win Probability
echo   ─────────────────────────────────────
echo.

REM ── Stop action ──────────────────────────────────────────────────────────
if "%ACTION%"=="stop" (
    echo   Stopping containers...
    docker compose down 2>nul || docker-compose down 2>nul
    echo   [OK] Containers stopped.
    goto :eof
)

REM ── Status action ────────────────────────────────────────────────────────
if "%ACTION%"=="status" (
    echo   Container status:
    docker compose ps 2>nul || docker-compose ps 2>nul
    echo.
    curl -sf "http://localhost:%PORT%/api/version" >nul 2>&1
    if !errorlevel! equ 0 (
        echo   [OK] Server is running at http://localhost:%PORT%
    ) else (
        echo   [WARN] Server is not responding on port %PORT%.
    )
    goto :eof
)

REM ── Start action ─────────────────────────────────────────────────────────

REM Step 1: Check Docker
echo   Step 1/5 — Checking prerequisites
echo.

where docker >nul 2>&1
if errorlevel 1 (
    echo   [ERROR] Docker is not installed.
    echo.
    echo   Please install Docker Desktop for Windows:
    echo     https://docs.docker.com/desktop/install/windows-install/
    echo.
    echo   After installation, restart this script.
    exit /b 1
)
echo   [OK] Docker is installed.

docker info >nul 2>&1
if errorlevel 1 (
    echo   [ERROR] Docker daemon is not running.
    echo.
    echo   Please start Docker Desktop from the Start menu or system tray.
    echo   Wait for it to finish starting, then run this script again.
    exit /b 1
)
echo   [OK] Docker daemon is running.

REM Check docker compose
docker compose version >nul 2>&1
if errorlevel 1 (
    docker-compose version >nul 2>&1
    if errorlevel 1 (
        echo   [ERROR] Docker Compose is not available.
        echo   Please update Docker Desktop to the latest version.
        exit /b 1
    )
    set "COMPOSE_CMD=docker-compose"
) else (
    set "COMPOSE_CMD=docker compose"
)
echo   [OK] Docker Compose is available.

REM Step 2: Create directories
echo.
echo   Step 2/5 — Creating directories

if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"
if not exist "data\models" mkdir "data\models"
if not exist "logs" mkdir "logs"
echo   [OK] data\ and logs\ directories ready.

REM Step 3: Configure environment
echo.
echo   Step 3/5 — Configuring environment

set "GIT_COMMIT=unknown"
where git >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%i in ('git rev-parse --short=8 HEAD 2^>nul') do set "GIT_COMMIT=%%i"
)

echo   [OK] Model type: %MODEL%
echo   [OK] Host port:  %PORT%
echo   [OK] Git commit: %GIT_COMMIT%

REM Step 4: Build Docker image
echo.
echo   Step 4/5 — Building Docker image

if "%REBUILD%"=="true" (
    echo   Forcing rebuild...
    %COMPOSE_CMD% build --no-cache --build-arg GIT_COMMIT=%GIT_COMMIT%
) else (
    %COMPOSE_CMD% build --build-arg GIT_COMMIT=%GIT_COMMIT%
)

if errorlevel 1 (
    echo   [ERROR] Docker build failed. Check the output above for details.
    exit /b 1
)
echo   [OK] Docker image built successfully.

REM Step 5: Start containers
echo.
echo   Step 5/5 — Starting server

%COMPOSE_CMD% up -d

if errorlevel 1 (
    echo   [ERROR] Failed to start containers. Check Docker Desktop for issues.
    exit /b 1
)

echo.
echo   [OK] Server starting at http://localhost:%PORT%
echo.
echo   Useful commands:
echo     %COMPOSE_CMD% logs -f         Follow server logs
echo     %~nx0 --status               Check server status
echo     %~nx0 --stop                 Stop the server
echo     %~nx0 --model xgboost        Restart with a different model
echo.

REM Wait for server
echo   Waiting for server to become healthy...
echo   (This may take a few minutes on first run while data is ingested.)
echo.

set /a ELAPSED=0
set /a MAX_WAIT=120

:wait_loop
if %ELAPSED% geq %MAX_WAIT% goto :wait_timeout
curl -sf "http://localhost:%PORT%/api/version" >nul 2>&1
if not errorlevel 1 (
    echo.
    echo   [OK] Server is healthy and ready!
    echo.
    echo   Open your browser: http://localhost:%PORT%
    echo.
    goto :eof
)
timeout /t 5 /nobreak >nul
set /a ELAPSED+=5
echo   Waiting... (%ELAPSED%s / %MAX_WAIT%s)
goto :wait_loop

:wait_timeout
echo.
echo   [WARN] Server hasn't responded within %MAX_WAIT%s.
echo   On first run, the initial data ingestion can take much longer.
echo   Check progress with: %COMPOSE_CMD% logs -f
echo.
echo   The server will be available at http://localhost:%PORT%
echo   once the initial setup completes.
echo.
goto :eof

:show_help
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo   --stop          Stop running containers
echo   --status        Show container status
echo   --rebuild       Force rebuild the Docker image
echo   --model TYPE    Model type: logistic^|lightgbm^|xgboost^|catboost^|mlp^|stacked (default: stacked)
echo   --port PORT     Host port (default: 30087)
echo   -h, --help      Show this help
goto :eof
