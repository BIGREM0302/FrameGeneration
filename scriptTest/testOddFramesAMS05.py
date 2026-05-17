from yuvProc import *

####################################################################################
# for test case 2
####################################################################################
orgFile = '../orgYUV/odd_H2_H3_AMS05_3840x2160_10bit_420_HLG.yuv'

#yuvFileListAfter = ['../bitstream/upscaled/odd_H2_H3_AMS05_27_0_5_up.layer0.yuv', '../bitstream/upscaled/odd_H2_H3_AMS05_32_0_5_up.layer0.yuv', '../bitstream/upscaled/odd_H2_H3_AMS05_37_0_5_up.layer0.yuv', '../bitstream/upscaled/odd_H2_H3_AMS05_42_0_5_up.layer0.yuv']   # yuv files after processing, from best quality to worst quality
yuvFileListAfter = [
    '../bitstream/generated/odd_H2_H3_AMS05_27_0_5_gen.layer0.yuv', 
    '../bitstream/generated/odd_H2_H3_AMS05_32_0_5_gen.layer0.yuv', 
    '../bitstream/generated/odd_H2_H3_AMS05_37_0_5_gen.layer0.yuv', 
    '../bitstream/generated/odd_H2_H3_AMS05_42_0_5_gen.layer0.yuv'
]

frameCount = 49 # (total frame number / 2) - 1
videoRateReference = [19418.8432, 7611.1632, 3055.2704, 1149.852]        # kbps, copy values from Excel file
videoPSNRBefore = [33.354902657018094, 32.47356390973943, 31.410535115309045, 30.43038560737588]    # pre-calculated PSNR

####################################################################################

# test case 1, basic BDrate calculation, just simple examples

####################################################################################
print('case 1:')
anchorRate = [9619.6875, 2327.2125, 1266.7425, 752.3925]
anchorPSNR = [43.9141, 42.1190, 41.2022, 40.1110]
testRate = [10172.8875, 2699.9325, 1479.57, 867.435]
testPSNR = [43.9295, 42.1153, 41.1366, 39.9195]

valBDrate = bd_rate(anchorRate, anchorPSNR, testRate, testPSNR)

print('  BDrate should be 15.851%')
print(f'  Calculated BDrate = {valBDrate:0.3f}%')

anchorRate = [9619.6875, 2327.2125, 1266.7425, 752.3925]
anchorPSNR = [43.9141, 42.1190, 41.2022, 40.1110]
testRate = anchorRate
testPSNR = [x + 0.1 for x in anchorPSNR] # assume PSNR improved

valBDrate = bd_rate(anchorRate, anchorPSNR, testRate, testPSNR)

print('\n  Fake PSNR improvement:')
print(f'  Calculated BDrate = {valBDrate:0.3f}%')


####################################################################################

# test case 2, use this case to calculate your BD rate

####################################################################################

print('\ncase 2:')

width = 3840
height = 2160
bytesPerPel = 2

print(f'  PSNR before: {videoPSNRBefore}')


videoPSNRAfter = []

for f in range(4):
  yuvFile = yuvFileListAfter[f]
  psnrYList = []
  psnrUList = []
  psnrVList = []

  for i in range(frameCount):
    frameOrg = getOneFrame(orgFile, width, height, bytesPerPel, i)
    frameCmp = getOneFrame(yuvFile, width, height, bytesPerPel, i)
    psnrY = psnr10(frameOrg['y'], frameCmp['y'])
    psnrU = psnr10(frameOrg['u'], frameCmp['u'])
    psnrV = psnr10(frameOrg['v'], frameCmp['v'])

    psnrYList.append(psnrY)
    psnrUList.append(psnrU)
    psnrVList.append(psnrV)

  averageY = sum(psnrYList) / len(psnrYList)
  averageU = sum(psnrUList) / len(psnrUList)
  averageV = sum(psnrVList) / len(psnrVList)

  averageYUV = getAveragePSNR(averageY, averageU, averageV)
  videoPSNRAfter.append(averageYUV)

print(f'  PSNR after : {videoPSNRAfter}')

videoPSNRReference = videoPSNRBefore                  # dB
videoRate = videoRateReference                        # kbps, assume no rate change
videoPSNR = videoPSNRAfter
valBDrate = bd_rate(videoRateReference, videoPSNRReference, videoRate, videoPSNR)

print(f'  Calculated BDrate = {valBDrate:0.3f}%')

