This is the Official Documentation for the Ornimetrics system

inorder to run the device enter this command:
export QT_QPA_PLATFORM=xcb
python3 "path_to_detect" \
  --model "path_to_pytorch" \
  --source 0 --conf 0.45 --imgsz 320 \
  --cam_w 640 --cam_h 480 --fourcc MJPG \
  --max_fps 12 --every_n 1 \
  --trap_settings "path_to_trapsettings.json" \
  --db "https://ornimetrics-default-rtdb.firebaseio.com (optional if want to contribute to model database)" \
  --session_key "session_1" \
  --servo_channel 1 --servo_closed 5 --servo_open 60 \
  --servo_min_us 600 --servo_max_us 2400 \
  --servo_slew 240 \
  --log_fps_interval 20 --preview --preview_scale 0.6 --quiet
