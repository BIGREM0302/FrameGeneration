
../bin/DecoderAppStatic -b ZombieClimbing2_27_0_4.bin -o tmp.yuv -p 0 --UpscaledOutput=2 --UpscaledOutputWidth=3840 --UpscaledOutputHeight=2160
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i tmp.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../upscaled/odd_ZombieClimbing2_27_0_4_up.layer0.yuv
rm -rf tmp.layer0.yuv

../bin/DecoderAppStatic -b ZombieClimbing2_32_0_4.bin -o tmp.yuv -p 0 --UpscaledOutput=2 --UpscaledOutputWidth=3840 --UpscaledOutputHeight=2160
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i tmp.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../upscaled/odd_ZombieClimbing2_32_0_4_up.layer0.yuv
rm -rf tmp.layer0.yuv

../bin/DecoderAppStatic -b ZombieClimbing2_37_0_4.bin -o tmp.yuv -p 0 --UpscaledOutput=2 --UpscaledOutputWidth=3840 --UpscaledOutputHeight=2160
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i tmp.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../upscaled/odd_ZombieClimbing2_37_0_4_up.layer0.yuv
rm -rf tmp.layer0.yuv

../bin/DecoderAppStatic -b ZombieClimbing2_42_0_4.bin -o tmp.yuv -p 0 --UpscaledOutput=2 --UpscaledOutputWidth=3840 --UpscaledOutputHeight=2160
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i tmp.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../upscaled/odd_ZombieClimbing2_42_0_4_up.layer0.yuv
rm -rf tmp.layer0.yuv


../bin/DecoderAppStatic -b H2_WalkInPark_27_0_4.bin -o tmp.yuv -p 0 --UpscaledOutput=2 --UpscaledOutputWidth=3840 --UpscaledOutputHeight=2160
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i tmp.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../upscaled/odd_H2_WalkInPark_27_0_4_up.layer0.yuv
rm -rf tmp.layer0.yuv

../bin/DecoderAppStatic -b H2_WalkInPark_32_0_4.bin -o tmp.yuv -p 0 --UpscaledOutput=2 --UpscaledOutputWidth=3840 --UpscaledOutputHeight=2160
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i tmp.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../upscaled/odd_H2_WalkInPark_32_0_4_up.layer0.yuv
rm -rf tmp.layer0.yuv

../bin/DecoderAppStatic -b H2_WalkInPark_37_0_4.bin -o tmp.yuv -p 0 --UpscaledOutput=2 --UpscaledOutputWidth=3840 --UpscaledOutputHeight=2160
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i tmp.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../upscaled/odd_H2_WalkInPark_37_0_4_up.layer0.yuv
rm -rf tmp.layer0.yuv

../bin/DecoderAppStatic -b H2_WalkInPark_42_0_4.bin -o tmp.yuv -p 0 --UpscaledOutput=2 --UpscaledOutputWidth=3840 --UpscaledOutputHeight=2160
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i tmp.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../upscaled/odd_H2_WalkInPark_42_0_4_up.layer0.yuv
rm -rf tmp.layer0.yuv



../bin/DecoderAppStatic -b Procession_25_0_4.bin -o tmp.yuv -p 0 --UpscaledOutput=2 --UpscaledOutputWidth=3840 --UpscaledOutputHeight=2160
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i tmp.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../upscaled/odd_Procession_25_0_4_up.layer0.yuv
rm -rf tmp.layer0.yuv

../bin/DecoderAppStatic -b Procession_30_0_4.bin -o tmp.yuv -p 0 --UpscaledOutput=2 --UpscaledOutputWidth=3840 --UpscaledOutputHeight=2160
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i tmp.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../upscaled/odd_Procession_30_0_4_up.layer0.yuv
rm -rf tmp.layer0.yuv

../bin/DecoderAppStatic -b Procession_35_0_4.bin -o tmp.yuv -p 0 --UpscaledOutput=2 --UpscaledOutputWidth=3840 --UpscaledOutputHeight=2160
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i tmp.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../upscaled/odd_Procession_35_0_4_up.layer0.yuv
rm -rf tmp.layer0.yuv

../bin/DecoderAppStatic -b Procession_40_0_4.bin -o tmp.yuv -p 0 --UpscaledOutput=2 --UpscaledOutputWidth=3840 --UpscaledOutputHeight=2160
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i tmp.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../upscaled/odd_Procession_40_0_4_up.layer0.yuv
rm -rf tmp.layer0.yuv



../bin/DecoderAppStatic -b H2_H3_AMS05_27_0_5.bin -o tmp.yuv -p 0 --UpscaledOutput=2 --UpscaledOutputWidth=3840 --UpscaledOutputHeight=2160
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i tmp.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../upscaled/odd_H2_H3_AMS05_27_0_5_up.layer0.yuv
rm -rf tmp.layer0.yuv

../bin/DecoderAppStatic -b H2_H3_AMS05_32_0_5.bin -o tmp.yuv -p 0 --UpscaledOutput=2 --UpscaledOutputWidth=3840 --UpscaledOutputHeight=2160
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i tmp.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../upscaled/odd_H2_H3_AMS05_32_0_5_up.layer0.yuv
rm -rf tmp.layer0.yuv

../bin/DecoderAppStatic -b H2_H3_AMS05_37_0_5.bin -o tmp.yuv -p 0 --UpscaledOutput=2 --UpscaledOutputWidth=3840 --UpscaledOutputHeight=2160
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i tmp.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../upscaled/odd_H2_H3_AMS05_37_0_5_up.layer0.yuv
rm -rf tmp.layer0.yuv

../bin/DecoderAppStatic -b H2_H3_AMS05_42_0_5.bin -o tmp.yuv -p 0 --UpscaledOutput=2 --UpscaledOutputWidth=3840 --UpscaledOutputHeight=2160
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i tmp.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../upscaled/odd_H2_H3_AMS05_42_0_5_up.layer0.yuv
rm -rf tmp.layer0.yuv



