@echo off
setlocal
set "PLATFORM="
set /p "PLATFORM=Platform (youtube, tiktok, instagram, netflix, spotify, twitch): "
set /p "FORMAT=Format (short_video or full_video): "
set /p "STATE=Capture state (e.g. playing_controls_visible): "
echo Open the target content on the device. After the runner activates it, you have four seconds to show the controls.
pause
pymobiledevice3 developer dvt xcuitest com.grozziie.media-collector.capture.xctrunner --target-bundle-id com.grozziie.media-collector --env "PLATFORM_NAME_INPUT=%PLATFORM%" --env "CONTENT_FORMAT=%FORMAT%" --env "CAPTURE_STATE=%STATE%" --output-log capture-log.txt
type capture-log.txt
pause
