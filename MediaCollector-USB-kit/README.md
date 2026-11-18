# Media Collector

## Prepare each iPhone or iPad on the Mac

Open MediaCollector.xcodeproj. Select your signing team on MediaCollector and Capture. The runner transfers each capture to the foreground companion over device loopback. Enable Developer Mode, trust the Mac, install YouTube and open the desired player. Choose the device and run Product → Test using the MediaCollector scheme. This installs the companion and XCTest runner and performs the first YouTube capture. Keep the development signing profiles current.

## Collect over USB on Windows

1. Install Python and Apple Mobile Device Service (included with iTunes for Windows). Install Apple Devices for file export. From a terminal run `py -m pip install -U pymobiledevice3`. Confirm `pymobiledevice3 developer dvt xcuitest --help` lists --env. Make sure the Python Scripts directory is on PATH.
2. Connect one prepared, unlocked device, trust the PC and run `pymobiledevice3 usbmux list`. Run `pymobiledevice3 mounter auto-mount` if the developer image is missing. For iOS 17.0–17.3 start `pymobiledevice3 remote tunneld` in a separate administrator terminal. See [USB installation and version-specific setup](https://github.com/doronz88/pymobiledevice3/blob/master/docs/installation.md).
3. Open the video in full-screen app mode. Double-click Capture.cmd per screenshot, enter the requested settings and use the four-second delay to reveal the desired controls. Keep the phone unlocked. Check the printed test status and the companion's import status. If interrupted after capture, open MediaCollector and tap Import captures.
4. Open Apple Devices → Files → MediaCollector, select data and click Save. Keep the directory tree intact. [Apple USB File Sharing](https://support.apple.com/en-us/120402)

For custom apps, run the same xcuitest command as Capture.cmd with extra --env TARGET_BUNDLE=your.app and a filesystem-safe platform name. Optional THEME and CONTENT_URL variables populate otherwise unknown fields. Inspect the PNG against raw_annotations before training. English labels, accessibility frame sizes, animated controls, protected playback, and estimated playheads require review. Full-screen single-app layouts are supported. The two curation files are initialized on first import and preserved thereafter.
