# Clipforu

A personal video clipper for GitHub Codespaces. It transcribes a video with Faster Whisper, suggests sentence-aligned 20–60 second moments, previews them, and exports vertical MP4 clips with optional word-group captions.

## Run it in GitHub Codespaces

1. Put these project files in the root of your GitHub repository, including `.devcontainer/`.
2. In GitHub, open the repository and choose **Code → Codespaces → Create codespace**. If a Codespace already exists, run **Codespaces: Rebuild Container** once so FFmpeg and Python packages are installed.
3. The app starts on port 5050. Open the **Ports** panel and open **Private Clipforu**. Keep the port private.
4. Upload a video. Choose English or Hungarian to skip language detection, or leave automatic detection selected. The first analysis downloads the model; later videos reuse it.
5. Preview the suggestions, adjust the start/end and crop position, then export. Use the **Download** link to save each clip to your own device.
6. Click **Delete source video and transcript** when finished. This removes the uploaded source and transcript data from the Codespace. If you forget, inactive job data expires after six hours while the app is running; expired work folders are also cleared the next time the app starts.

## Storage and privacy

This runs on a GitHub-hosted Codespace, not on your Mac. Uploading sends the video to that Codespace. Port forwarding is configured as private and GitHub authentication is required to access it. The source is kept in the Codespace only for preview and export. Exported clips are saved in the repository workspace's ignored `clips/` folder; download them and delete them with the per-clip **Delete** button when done. These media folders are excluded from Git commits.

Codespaces has finite storage and included usage is limited by your GitHub account and plan. This project cannot guarantee unlimited free processing or storage. Large videos use Codespace disk and CPU; the app checks free disk space and rejects uploads above the configured limit. The default upload cap is 20 GB and can be changed with `MAX_UPLOAD_GB`. The clipper does not upload media to a separate AI API, but downloading the speech model on first use requires network access.

Stopping or deleting the Codespace stops the app. Files in its workspace persist across ordinary stops, but deleting the Codespace deletes its unpushed workspace data. Download clips you want to keep before deleting the Codespace.

## What the suggestions do

Faster Whisper creates a timestamped transcript. The app ranks sentence-aligned windows using transcript cues such as questions, selected hook words, complete sentence endings, speech density, and a preferred duration near 40 seconds. This is a local, explainable first pass; it does not understand story context, detect visual action, track a speaker, or know whether a quote is misleading without surrounding context. Preview and edit candidates before posting.

## Settings

Set these environment variables in `.devcontainer/devcontainer.json` before rebuilding:

| Variable | Default | Purpose |
| --- | --- | --- |
| `WHISPER_MODEL` | `base` | Faster/smaller `base`, or a larger model for improved accuracy at higher CPU and memory cost. |
| `MAX_UPLOAD_GB` | `20` | Maximum accepted upload size. |
| `JOB_TTL_HOURS` | `6` | How long inactive uploaded jobs are retained before cleanup. |
| `CLIPS_DIR` | `./clips` | Where exported MP4 clips are stored. |
| `CLIP_WORK_DIR` | `./work` | Temporary source and transcript files. |

The application binds to port 5050. Do not change the forwarded port to public for a personal library.
