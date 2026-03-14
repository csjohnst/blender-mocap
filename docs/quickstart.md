# Quick Start Guide — Blender Motion Capture

## Prerequisites

- **Blender 4.5 LTS** (installed on Arch Linux)
- **System Python 3.10–3.12** (required by MediaPipe)
- **Webcam** accessible at `/dev/video*`
- **PortAudio** for audio capture: `sudo pacman -S portaudio`

## Installation

1. Build the addon zip (or use a pre-built one):

   ```bash
   cd blender-mocap
   ./scripts/build_addon.sh
   ```

2. Open Blender and go to **Edit → Preferences → Add-ons**

3. Click **Install** and select `dist/blender_mocap.zip`

4. Enable the **"Motion Capture"** addon by checking the box

5. On first enable, the addon automatically creates a Python virtual environment at `~/.blender-mocap/venv/` and installs dependencies (MediaPipe, OpenCV, sounddevice, numpy). This may take a minute.

## Setting Up Your Scene

1. **Add a Rigify armature:**
   - `Add → Armature → Human (Meta-Rig)`
   - Select the metarig, go to the **Armature** properties tab
   - Click **Generate Rig** (requires the Rigify addon to be enabled in Preferences)

2. **Open the Motion Capture panel:**
   - In the 3D Viewport, press `N` to open the sidebar
   - Click the **"Motion Capture"** tab

## Configuring the Panel

The panel has four sections:

### Setup

| Setting | Description |
|---------|-------------|
| **Camera** | Select your webcam from the dropdown |
| **Armature** | Pick your generated Rigify rig (only Rigify armatures with `rig_id` appear) |
| **Audio Source** | Choose a microphone (webcam mic, USB mic, or system default) |
| **Smoothing** | Controls real-time jitter reduction (0 = none, 1 = heavy). Start around **0.3** |

### Capture

Three buttons control the capture workflow:

- **Start Preview** — opens a separate OpenCV window showing your webcam with a green skeleton overlay. Your Rigify rig moves in the viewport in real-time.
- **Record** — begins recording motion and audio (only available during preview)
- **Stop** — stops recording/preview. If recording, the captured motion is baked into a Blender Action.

### Recordings

Lists all captured recordings with frame counts. For each recording you can:

- **Smooth** — applies post-capture smoothing to reduce any remaining jitter
- **Delete** — removes the recording and its associated audio file

### Export

Export the selected recording in any format:

| Format | Use case |
|--------|----------|
| **.blend** | Append/link the Action into other Blender projects via File → Append |
| **FBX** | Import into other 3D software (Unity, Unreal, Maya, etc.) |
| **BVH** | Standard motion capture format, widely supported |
| **Audio (WAV)** | The synchronized audio recording |

When exporting motion (.blend/FBX/BVH), the audio WAV is automatically copied alongside it.

## Typical Workflow

```
1. Set up Rigify rig and select it in the panel
2. Click "Start Preview" — position yourself in front of the camera
3. Adjust Smoothing if the skeleton is too jittery or too sluggish
4. When ready, click "Record"
5. Perform your motion
6. Click "Stop" — motion is baked into an Action (e.g., MoCap_001)
7. Optionally click "Smooth" to refine the animation
8. Export in your preferred format
```

## Tips

- **Stand far enough back** so your full body is visible to the camera. MediaPipe needs to see your whole pose for accurate tracking.

- **Good lighting** dramatically improves pose detection accuracy. Avoid strong backlighting.

- **Start with the smoothing slider around 0.3.** Too low and you'll get jitter; too high and fast movements will feel delayed.

- **Use the preview** to check your camera angle and lighting before recording. The green skeleton overlay shows exactly what MediaPipe is detecting.

- **Post-capture smoothing** can be applied multiple times. Each click smooths the F-curves further.

- **Audio is recorded separately** as a WAV file stored at `~/.blender-mocap/recordings/`. You can import it into Blender's Video Sequence Editor or use it with your rendered output.

- **Multiple takes** are fine — each recording creates a new Action (MoCap_001, MoCap_002, etc.). You can keep or delete any of them.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No cameras found" | Check that your webcam is connected and accessible (`ls /dev/video*`) |
| Venv creation fails | Ensure `python3` (3.10–3.12) is installed: `python3 --version`. Install the `python-virtualenv` package if needed. |
| Capture server won't start | Check the terminal/console for errors. The server runs as a subprocess — stderr is captured. |
| Pose detection is poor | Improve lighting, stand further back, and ensure your full body is visible |
| Preview window not appearing | The OpenCV window opens as a separate system window — check your taskbar |
| Armature doesn't appear in picker | Make sure you clicked "Generate Rig" on the metarig. Only generated Rigify rigs (with `rig_id`) are listed. |
| Animation looks stiff | Increase the Smoothing slider, or apply post-capture Smooth to the recording |
| No audio recorded | Check that the correct Audio Source is selected and that PortAudio is installed |

## File Locations

| Path | Contents |
|------|----------|
| `~/.blender-mocap/venv/` | Python virtual environment with dependencies |
| `~/.blender-mocap/recordings/` | Audio WAV files from recording sessions |
| `/tmp/blender-mocap-*.sock` | Unix socket (auto-cleaned on stop) |
