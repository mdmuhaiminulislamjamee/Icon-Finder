**Use `Capture.cmd` on the Windows PC to take a screenshot.** The collector and capture runner are already installed on this iPad.

**One-time Windows setup**

1. Download the [USB collector kit](/private/tmp/MediaCollector-USB-kit.zip), copy it to the Windows PC, and extract it.
2. Install Python and Apple Mobile Device Service on Windows. The service comes with iTunes for Windows. Install **Apple Devices** to copy the screenshots afterward.
3. Connect the iPad by USB, unlock it, and tap **Trust** if prompted. In Command Prompt, run:

   ```bat
   py -m pip install -U pymobiledevice3
   pymobiledevice3 usbmux list
   ```

   Confirm the iPad appears. If Windows says `pymobiledevice3` is unrecognized, add Python’s `Scripts` folder to `PATH` and reopen Command Prompt.

**For each screenshot**

1. Open the video you want on the iPad.
2. Double-click **`Capture.cmd`** on Windows. For a playing YouTube Short with controls visible, enter `youtube`, then `short_video`, then `playing_controls_visible`.
3. **Press any key on the PC to start.** When YouTube comes forward on the iPad, you have **four seconds** to show the controls you want captured.
4. Look for **“Dataset ready”** on the iPad. To copy the PNGs to your PC, open **Apple Devices → iPad → Files → MediaCollector**, select **data**, and click **Save**. You can take several screenshots before copying them.

The Windows USB capture workflow has not yet been verified end to end on this iPad.