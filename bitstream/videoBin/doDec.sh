
../bin/DecoderAppStatic -b ZombieClimbing2_27_0_4.bin -o ZombieClimbing2_27_0_4.yuv -p 0
../bin/DecoderAppStatic -b ZombieClimbing2_27_0_4.bin -o ZombieClimbing2_27_0_4.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 1920x1080 -i ZombieClimbing2_27_0_4.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../base/odd_ZombieClimbing2_27_0_4.layer0.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i ZombieClimbing2_27_0_4.layer1.yuv -vf "select='eq(mod(n\,2)\,0)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../enhance/even_ZombieClimbing2_27_0_4.layer1.yuv
rm -rf ZombieClimbing2_27*.yuv

../bin/DecoderAppStatic -b ZombieClimbing2_32_0_4.bin -o ZombieClimbing2_32_0_4.yuv -p 0
../bin/DecoderAppStatic -b ZombieClimbing2_32_0_4.bin -o ZombieClimbing2_32_0_4.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 1920x1080 -i ZombieClimbing2_32_0_4.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../base/odd_ZombieClimbing2_32_0_4.layer0.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i ZombieClimbing2_32_0_4.layer1.yuv -vf "select='eq(mod(n\,2)\,0)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../enhance/even_ZombieClimbing2_32_0_4.layer1.yuv
rm -rf ZombieClimbing2_32*.yuv

../bin/DecoderAppStatic -b ZombieClimbing2_37_0_4.bin -o ZombieClimbing2_37_0_4.yuv -p 0
../bin/DecoderAppStatic -b ZombieClimbing2_37_0_4.bin -o ZombieClimbing2_37_0_4.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 1920x1080 -i ZombieClimbing2_37_0_4.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../base/odd_ZombieClimbing2_37_0_4.layer0.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i ZombieClimbing2_37_0_4.layer1.yuv -vf "select='eq(mod(n\,2)\,0)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../enhance/even_ZombieClimbing2_37_0_4.layer1.yuv
rm -rf ZombieClimbing2_37*.yuv

../bin/DecoderAppStatic -b ZombieClimbing2_42_0_4.bin -o ZombieClimbing2_42_0_4.yuv -p 0
../bin/DecoderAppStatic -b ZombieClimbing2_42_0_4.bin -o ZombieClimbing2_42_0_4.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 1920x1080 -i ZombieClimbing2_42_0_4.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../base/odd_ZombieClimbing2_42_0_4.layer0.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i ZombieClimbing2_42_0_4.layer1.yuv -vf "select='eq(mod(n\,2)\,0)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../enhance/even_ZombieClimbing2_42_0_4.layer1.yuv
rm -rf ZombieClimbing2_42*.yuv



../bin/DecoderAppStatic -b Procession_25_0_4.bin -o Procession_25_0_4.yuv -p 0
../bin/DecoderAppStatic -b Procession_25_0_4.bin -o Procession_25_0_4.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 1920x1080 -i Procession_25_0_4.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../base/odd_Procession_25_0_4.layer0.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i Procession_25_0_4.layer1.yuv -vf "select='eq(mod(n\,2)\,0)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../enhance/even_Procession_25_0_4.layer1.yuv
rm -rf Procession_25*.yuv

../bin/DecoderAppStatic -b Procession_30_0_4.bin -o Procession_30_0_4.yuv -p 0
../bin/DecoderAppStatic -b Procession_30_0_4.bin -o Procession_30_0_4.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 1920x1080 -i Procession_30_0_4.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../base/odd_Procession_30_0_4.layer0.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i Procession_30_0_4.layer1.yuv -vf "select='eq(mod(n\,2)\,0)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../enhance/even_Procession_30_0_4.layer1.yuv
rm -rf Procession_30*.yuv

../bin/DecoderAppStatic -b Procession_35_0_4.bin -o Procession_35_0_4.yuv -p 0
../bin/DecoderAppStatic -b Procession_35_0_4.bin -o Procession_35_0_4.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 1920x1080 -i Procession_35_0_4.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../base/odd_Procession_35_0_4.layer0.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i Procession_35_0_4.layer1.yuv -vf "select='eq(mod(n\,2)\,0)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../enhance/even_Procession_35_0_4.layer1.yuv
rm -rf Procession_35*.yuv

../bin/DecoderAppStatic -b Procession_40_0_4.bin -o Procession_40_0_4.yuv -p 0
../bin/DecoderAppStatic -b Procession_40_0_4.bin -o Procession_40_0_4.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 1920x1080 -i Procession_40_0_4.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../base/odd_Procession_40_0_4.layer0.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i Procession_40_0_4.layer1.yuv -vf "select='eq(mod(n\,2)\,0)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../enhance/even_Procession_40_0_4.layer1.yuv
rm -rf Procession_40*.yuv



../bin/DecoderAppStatic -b H2_WalkInPark_27_0_4.bin -o H2_WalkInPark_27_0_4.yuv -p 0
../bin/DecoderAppStatic -b H2_WalkInPark_27_0_4.bin -o H2_WalkInPark_27_0_4.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 1920x1080 -i H2_WalkInPark_27_0_4.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../base/odd_H2_WalkInPark_27_0_4.layer0.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i H2_WalkInPark_27_0_4.layer1.yuv -vf "select='eq(mod(n\,2)\,0)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../enhance/even_H2_WalkInPark_27_0_4.layer1.yuv
rm -rf H2_WalkInPark_27*.yuv

../bin/DecoderAppStatic -b H2_WalkInPark_32_0_4.bin -o H2_WalkInPark_32_0_4.yuv -p 0
../bin/DecoderAppStatic -b H2_WalkInPark_32_0_4.bin -o H2_WalkInPark_32_0_4.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 1920x1080 -i H2_WalkInPark_32_0_4.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../base/odd_H2_WalkInPark_32_0_4.layer0.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i H2_WalkInPark_32_0_4.layer1.yuv -vf "select='eq(mod(n\,2)\,0)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../enhance/even_H2_WalkInPark_32_0_4.layer1.yuv
rm -rf H2_WalkInPark_32*.yuv

../bin/DecoderAppStatic -b H2_WalkInPark_37_0_4.bin -o H2_WalkInPark_37_0_4.yuv -p 0
../bin/DecoderAppStatic -b H2_WalkInPark_37_0_4.bin -o H2_WalkInPark_37_0_4.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 1920x1080 -i H2_WalkInPark_37_0_4.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../base/odd_H2_WalkInPark_37_0_4.layer0.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i H2_WalkInPark_37_0_4.layer1.yuv -vf "select='eq(mod(n\,2)\,0)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../enhance/even_H2_WalkInPark_37_0_4.layer1.yuv
rm -rf H2_WalkInPark_37*.yuv

../bin/DecoderAppStatic -b H2_WalkInPark_42_0_4.bin -o H2_WalkInPark_42_0_4.yuv -p 0
../bin/DecoderAppStatic -b H2_WalkInPark_42_0_4.bin -o H2_WalkInPark_42_0_4.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 1920x1080 -i H2_WalkInPark_42_0_4.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../base/odd_H2_WalkInPark_42_0_4.layer0.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i H2_WalkInPark_42_0_4.layer1.yuv -vf "select='eq(mod(n\,2)\,0)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../enhance/even_H2_WalkInPark_42_0_4.layer1.yuv
rm -rf H2_WalkInPark_42*.yuv



../bin/DecoderAppStatic -b H2_H3_AMS05_27_0_5.bin -o H2_H3_AMS05_27_0_5.yuv -p 0
../bin/DecoderAppStatic -b H2_H3_AMS05_27_0_5.bin -o H2_H3_AMS05_27_0_5.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 1920x1080 -i H2_H3_AMS05_27_0_5.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../base/odd_H2_H3_AMS05_27_0_5.layer0.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i H2_H3_AMS05_27_0_5.layer1.yuv -vf "select='eq(mod(n\,2)\,0)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../enhance/even_H2_H3_AMS05_27_0_5.layer1.yuv
rm -rf H2_H3_AMS05_27*.yuv

../bin/DecoderAppStatic -b H2_H3_AMS05_32_0_5.bin -o H2_H3_AMS05_32_0_5.yuv -p 0
../bin/DecoderAppStatic -b H2_H3_AMS05_32_0_5.bin -o H2_H3_AMS05_32_0_5.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 1920x1080 -i H2_H3_AMS05_32_0_5.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../base/odd_H2_H3_AMS05_32_0_5.layer0.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i H2_H3_AMS05_32_0_5.layer1.yuv -vf "select='eq(mod(n\,2)\,0)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../enhance/even_H2_H3_AMS05_32_0_5.layer1.yuv
rm -rf H2_H3_AMS05_32*.yuv

../bin/DecoderAppStatic -b H2_H3_AMS05_37_0_5.bin -o H2_H3_AMS05_37_0_5.yuv -p 0
../bin/DecoderAppStatic -b H2_H3_AMS05_37_0_5.bin -o H2_H3_AMS05_37_0_5.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 1920x1080 -i H2_H3_AMS05_37_0_5.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../base/odd_H2_H3_AMS05_37_0_5.layer0.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i H2_H3_AMS05_37_0_5.layer1.yuv -vf "select='eq(mod(n\,2)\,0)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../enhance/even_H2_H3_AMS05_37_0_5.layer1.yuv
rm -rf H2_H3_AMS05_37*.yuv

../bin/DecoderAppStatic -b H2_H3_AMS05_42_0_5.bin -o H2_H3_AMS05_42_0_5.yuv -p 0
../bin/DecoderAppStatic -b H2_H3_AMS05_42_0_5.bin -o H2_H3_AMS05_42_0_5.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 1920x1080 -i H2_H3_AMS05_42_0_5.layer0.yuv -vf "select='eq(mod(n\,2)\,1)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../base/odd_H2_H3_AMS05_42_0_5.layer0.yuv
ffmpeg -y -f rawvideo -pix_fmt yuv420p10le -s:v 3840x2160 -i H2_H3_AMS05_42_0_5.layer1.yuv -vf "select='eq(mod(n\,2)\,0)'" -vsync 0 -f rawvideo -pix_fmt yuv420p10le -frames:v 50 ../enhance/even_H2_H3_AMS05_42_0_5.layer1.yuv
rm -rf H2_H3_AMS05_42*.yuv


