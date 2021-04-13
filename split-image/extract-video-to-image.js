var {spawn} = require('child_process');
const path = require('path');
c_process = spawn("ffmpeg", ["-i", "as.mp4", "aaa%04d.png", "-hide_banner"],{
    stdio: ["pipe", "pipe", "pipe"]
  }); 