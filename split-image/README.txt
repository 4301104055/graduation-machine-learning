command to run: node extract-video-to-image.js

nhớ để ffmpeg.exe vô cùng folder

spawn("ffmpeg", ["-i", "as.mp4", "aaa%04d.png", "-hide_banner"]

ffmpeg là tên file ffmpeg.exe

những cái trong ngoặc [] là từng parameter. sau này có thể tạo 1 hàm với nhiều tham số truyền vào để thay đổi tên video, định dạng và tên file xuất, định đạng.

Example:
Function SplitFrame(nameVideo, extensionVideo, nameImage, extensionImage){
	c_process = spawn("ffmpeg", ["-i", nameVideo + "." + extensionVideo,  nameImage + "." + extensionImage,"-hide_banner"],{
    stdio: ["pipe", "pipe", "pipe"]
  }); 
}

=> Ý tưởng thôi nên từ từ động tới rồi thay đổi sau :3 <3