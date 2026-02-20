NuWave Video Generator that lets you create videos from an audio file and a cover image. It's designed for quick processing, especially for longer audio files.

Here's a quick rundown of what it does:

Input: You provide an audio file (MP3, WAV, FLAC, M4A, etc.) and a cover image (JPG, PNG).
Video Generation:
It uses FFmpeg to render a scrolling audio waveform directly from your audio and composites it over a background.
Optionally, if your system has CUDA and the necessary libraries (like diffusers for LTX-Video AI), it can generate an AI-animated background video from your image. Otherwise, it uses a static image background.
The colors and theme for the video are automatically derived from your cover image.
User Interface: It uses Gradio to provide a web-based interface where you can upload your files, generate the video, and download the final MP4. It also displays system info like RAM, CPU, and GPU usage.
Performance: A key feature is its optimized workflow that renders the waveform entirely within FFmpeg, making it much faster than traditional Python-based frame-by-frame rendering.
Essentially, it's a tool to easily create visually engaging audio-reactive videos, perfect for sharing music or podcasts with a dynamic visual element.
