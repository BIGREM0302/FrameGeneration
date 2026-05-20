from yuvProc import *
import numpy as np

####################################################################################
# for test case 2
####################################################################################
orgFile = '../orgYUV/odd_H2_WalkInPark_3840x2160_10_60fps_HLG.yuv'

yuvFileListAfter = [
    '../bitstream/generated/odd_H2_WalkInPark_27_0_4_gen.layer0.yuv', 
    '../bitstream/generated/odd_H2_WalkInPark_32_0_4_gen.layer0.yuv', 
    '../bitstream/generated/odd_H2_WalkInPark_37_0_4_gen.layer0.yuv', 
    '../bitstream/generated/odd_H2_WalkInPark_42_0_4_gen.layer0.yuv'
]

qpList = [27, 32, 37, 42]

frameCount = 49
videoRateReference = [14634.8024, 6836.8248, 3211.4488, 1428.1816]
videoPSNRBefore = [35.88446626291224, 34.154505094684744, 32.31940736651056, 30.692211121619913]

####################################################################################
# test case 1
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
testPSNR = [x + 0.1 for x in anchorPSNR]

valBDrate = bd_rate(anchorRate, anchorPSNR, testRate, testPSNR)

print('\n  Fake PSNR improvement:')
print(f'  Calculated BDrate = {valBDrate:0.3f}%')


####################################################################################
# test case 2
####################################################################################

print('\ncase 2:')

width = 3840
height = 2160
bytesPerPel = 2

print(f'  PSNR before YUV: {videoPSNRBefore}')

videoPSNRAfterY = []
videoPSNRAfterU = []
videoPSNRAfterV = []
videoPSNRAfterYUV = []

all_qp_stats = []

for f in range(4):
    qp = qpList[f]
    yuvFile = yuvFileListAfter[f]

    psnrYList = []
    psnrUList = []
    psnrVList = []
    psnrYUVList = []

    print(f'\n================ QP {qp} ================')
    print(f'File: {yuvFile}')

    for i in range(frameCount):
        frameOrg = getOneFrame(orgFile, width, height, bytesPerPel, i)
        frameCmp = getOneFrame(yuvFile, width, height, bytesPerPel, i)

        psnrY = psnr10(frameOrg['y'], frameCmp['y'])
        psnrU = psnr10(frameOrg['u'], frameCmp['u'])
        psnrV = psnr10(frameOrg['v'], frameCmp['v'])
        psnrYUV = getAveragePSNR(psnrY, psnrU, psnrV)

        psnrYList.append(psnrY)
        psnrUList.append(psnrU)
        psnrVList.append(psnrV)
        psnrYUVList.append(psnrYUV)

    psnrYArr = np.array(psnrYList)
    psnrUArr = np.array(psnrUList)
    psnrVArr = np.array(psnrVList)
    psnrYUVArr = np.array(psnrYUVList)

    averageY = float(np.mean(psnrYArr))
    averageU = float(np.mean(psnrUArr))
    averageV = float(np.mean(psnrVArr))
    averageYUV = getAveragePSNR(averageY, averageU, averageV)

    videoPSNRAfterY.append(averageY)
    videoPSNRAfterU.append(averageU)
    videoPSNRAfterV.append(averageV)
    videoPSNRAfterYUV.append(averageYUV)

    # 找最差 frame
    worstYIdx = int(np.argmin(psnrYArr))
    worstUIdx = int(np.argmin(psnrUArr))
    worstVIdx = int(np.argmin(psnrVArr))
    worstYUVIdx = int(np.argmin(psnrYUVArr))

    # 統計資訊
    stats = {
        'qp': qp,
        'Y_mean': averageY,
        'U_mean': averageU,
        'V_mean': averageV,
        'YUV_mean': averageYUV,
        'Y_min': float(np.min(psnrYArr)),
        'U_min': float(np.min(psnrUArr)),
        'V_min': float(np.min(psnrVArr)),
        'YUV_min': float(np.min(psnrYUVArr)),
        'Y_std': float(np.std(psnrYArr)),
        'U_std': float(np.std(psnrUArr)),
        'V_std': float(np.std(psnrVArr)),
        'YUV_std': float(np.std(psnrYUVArr)),
        'worstYIdx': worstYIdx,
        'worstUIdx': worstUIdx,
        'worstVIdx': worstVIdx,
        'worstYUVIdx': worstYUVIdx,
    }

    all_qp_stats.append(stats)

    print('\n[Average PSNR]')
    print(f'  Y   : {averageY:8.4f} dB')
    print(f'  U   : {averageU:8.4f} dB')
    print(f'  V   : {averageV:8.4f} dB')
    print(f'  YUV : {averageYUV:8.4f} dB')

    print('\n[Frame PSNR distribution]')
    print(f'  Y   : min={np.min(psnrYArr):8.4f}, max={np.max(psnrYArr):8.4f}, std={np.std(psnrYArr):8.4f}')
    print(f'  U   : min={np.min(psnrUArr):8.4f}, max={np.max(psnrUArr):8.4f}, std={np.std(psnrUArr):8.4f}')
    print(f'  V   : min={np.min(psnrVArr):8.4f}, max={np.max(psnrVArr):8.4f}, std={np.std(psnrVArr):8.4f}')
    print(f'  YUV : min={np.min(psnrYUVArr):8.4f}, max={np.max(psnrYUVArr):8.4f}, std={np.std(psnrYUVArr):8.4f}')

    print('\n[Worst frames]')
    print(f'  Worst Y   frame: {worstYIdx:2d}, PSNR = {psnrYArr[worstYIdx]:8.4f} dB')
    print(f'  Worst U   frame: {worstUIdx:2d}, PSNR = {psnrUArr[worstUIdx]:8.4f} dB')
    print(f'  Worst V   frame: {worstVIdx:2d}, PSNR = {psnrVArr[worstVIdx]:8.4f} dB')
    print(f'  Worst YUV frame: {worstYUVIdx:2d}, PSNR = {psnrYUVArr[worstYUVIdx]:8.4f} dB')

    print('\n[Lowest 5 YUV frames]')
    lowest5 = np.argsort(psnrYUVArr)[:5]
    for idx in lowest5:
        print(
            f'  frame {idx:2d}: '
            f'Y={psnrYArr[idx]:8.4f}, '
            f'U={psnrUArr[idx]:8.4f}, '
            f'V={psnrVArr[idx]:8.4f}, '
            f'YUV={psnrYUVArr[idx]:8.4f}'
        )


####################################################################################
# Summary table
####################################################################################

print('\n\n================ Overall PSNR Summary ================')
print('QP | Rate(kbps) | Before_YUV | After_Y | After_U | After_V | After_YUV | Delta_YUV')
print('---|------------|------------|---------|---------|---------|-----------|----------')

for i in range(4):
    delta = videoPSNRAfterYUV[i] - videoPSNRBefore[i]
    print(
        f'{qpList[i]:2d} | '
        f'{videoRateReference[i]:10.4f} | '
        f'{videoPSNRBefore[i]:10.4f} | '
        f'{videoPSNRAfterY[i]:7.4f} | '
        f'{videoPSNRAfterU[i]:7.4f} | '
        f'{videoPSNRAfterV[i]:7.4f} | '
        f'{videoPSNRAfterYUV[i]:9.4f} | '
        f'{delta:+8.4f}'
    )


####################################################################################
# BD-rate
####################################################################################

videoPSNRReference = videoPSNRBefore
videoRate = videoRateReference
videoPSNR = videoPSNRAfterYUV

valBDrateYUV = bd_rate(videoRateReference, videoPSNRReference, videoRate, videoPSNR)

print('\n================ BD-rate Summary ================')
print(f'  Reference PSNR YUV : {videoPSNRReference}')
print(f'  Generated PSNR YUV : {videoPSNR}')
print(f'  Calculated BD-rate YUV = {valBDrateYUV:0.3f}%')

# 這裡注意：
# 你目前只有 videoPSNRBefore 是 YUV average，沒有 before 的 Y/U/V 分開資料。
# 所以 per-channel BD-rate 需要你提供 before 的 Y/U/V PSNR。
# 如果你有 before 的 Y/U/V，可以放進下面三個 list。

videoPSNRBeforeY = None
videoPSNRBeforeU = None
videoPSNRBeforeV = None

if videoPSNRBeforeY is not None:
    valBDrateY = bd_rate(videoRateReference, videoPSNRBeforeY, videoRate, videoPSNRAfterY)
    print(f'  Calculated BD-rate Y   = {valBDrateY:0.3f}%')

if videoPSNRBeforeU is not None:
    valBDrateU = bd_rate(videoRateReference, videoPSNRBeforeU, videoRate, videoPSNRAfterU)
    print(f'  Calculated BD-rate U   = {valBDrateU:0.3f}%')

if videoPSNRBeforeV is not None:
    valBDrateV = bd_rate(videoRateReference, videoPSNRBeforeV, videoRate, videoPSNRAfterV)
    print(f'  Calculated BD-rate V   = {valBDrateV:0.3f}%')


####################################################################################
# Diagnosis hints
####################################################################################

print('\n================ Diagnosis Hints ================')
