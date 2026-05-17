import os
import subprocess
import numpy as np
import math

PIX_MAX = 1020.0  # for PSNR10
PIX_MAX_8B = 255.0

#################################################################################################

def getOneFrame(yuvFile, width, height, bytesPerPel, frameNO, pixMult=1):
    
    oneFrame = {}
    
    # only support 420
    pixN_Y = width * height
    pixN_C = int((width * height)/4)
    pixN = pixN_Y + pixN_C * 2
    byteN_Y = pixN_Y * bytesPerPel
    byteN_C = pixN_C * bytesPerPel
    byteN = pixN * bytesPerPel
    
    dt = np.uint16 if bytesPerPel == 2 else np.uint8
    dt = np.dtype(dt)
    dt = dt.newbyteorder('<')
    
    with open(yuvFile, 'rb') as f_yuv:
        f_yuv.seek(frameNO * byteN)
        frameY = f_yuv.read(byteN_Y)
        frameY_np = np.frombuffer(frameY, dtype=dt)
        if bytesPerPel == 1:
            frameY_np = np.array(frameY_np, dtype=np.uint16)
            frameY_np = frameY_np * pixMult
        oneFrame['y'] = frameY_np
        
        frameC = f_yuv.read(byteN_C)
        frameC_np = np.frombuffer(frameC, dtype=dt)
        if bytesPerPel == 1:
            frameC_np = np.array(frameC_np, dtype=np.uint16)
            frameC_np = frameC_np * pixMult
        oneFrame['u'] = frameC_np
        
        frameC = f_yuv.read(byteN_C)
        frameC_np = np.frombuffer(frameC, dtype=dt)
        if bytesPerPel == 1:
            frameC_np = np.array(frameC_np, dtype=np.uint16)        
            frameC_np = frameC_np * pixMult
        oneFrame['v'] = frameC_np
        
    return oneFrame
        
def psnr10(original, compressed):
    ssd = np.sum((original.astype(np.int16) - compressed.astype(np.int16)) ** 2)
    mse = ssd / original.size
    
    if(mse == 0):
        return 0, 100
    
    max_pixel = PIX_MAX
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def psnr8(original, compressed):
    ssd = np.sum((original.astype(np.int16) - compressed.astype(np.int16)) ** 2)
    mse = ssd / original.size
    
    if(mse == 0):
        return 0, 100
    
    max_pixel = PIX_MAX_8B
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))    
    return psnr

def getAveragePSNR(psnr_y, psnr_u, psnr_v):
    psnr = (psnr_y * 6.0 + psnr_u + psnr_v) / 8.0
    return psnr

def saveOneFrameYUV(oneFrame, width, height, bytesPerPel, outYUVFileName, appendToYUV=False):

    oneFrameOut = oneFrame
    if len(oneFrameOut['y'].shape) != 2:
        oneFrameOut['y'] = oneFrame['y'].reshape([height, width])
        oneFrameOut['u'] = oneFrame['u'].reshape([height//2, width//2])
        oneFrameOut['v'] = oneFrame['v'].reshape([height//2, width//2])
    
    # only support 420
    with open(outYUVFileName, 'ab' if appendToYUV else 'wb') as fYuv:
        for y in range(height):
            for x in range(width):
                pix = oneFrameOut['y'][y, x]
                pix = pix.item()
                pix = pix.to_bytes(bytesPerPel, 'little')     
                fYuv.write(pix)
                
        for y in range(height//2):
            for x in range(width//2):
                pix = oneFrameOut['u'][y, x]
                pix = pix.item()
                pix = pix.to_bytes(bytesPerPel, 'little')     
                fYuv.write(pix)
                
        for y in range(height//2):
            for x in range(width//2):
                pix = oneFrameOut['v'][y, x]
                pix = pix.item()
                pix = pix.to_bytes(bytesPerPel, 'little')     
                fYuv.write(pix)                

def bd_rate(rate1, psnr1, rate2, psnr2, piecewise=False):
    return xbd_rate(np.array(rate1), np.array(psnr1), np.array(rate2), np.array(psnr2), piecewise)

def xbd_rate(rate1, psnr1, rate2, psnr2, piecewise=False):
    log_rate1 = np.log(rate1)
    log_rate2 = np.log(rate2)
    min_int = np.max([np.min(psnr1), np.min(psnr2)])
    max_int = np.min([np.max(psnr1), np.max(psnr2)])

    if piecewise:
        p1 = np.polyfit(psnr1, log_rate1, 3)
        p2 = np.polyfit(psnr2, log_rate2, 3)
        p1_int = np.polyint(p1)
        p2_int = np.polyint(p2)

        int1 = np.polyval(p1_int, max_int) - np.polyval(p1_int, min_int)
        int2 = np.polyval(p2_int, max_int) - np.polyval(p2_int, min_int)
        return (np.exp((int2 - int1) / (max_int - min_int)) - 1) * 100

    v1 = xbdrate_int(rate1, psnr1, min_int, max_int)
    v2 = xbdrate_int(rate2, psnr2, min_int, max_int)
    avg = (v2 - v1) / (max_int - min_int)
    return 100 * ((10**avg) - 1)


def xbdrate_int(rate, psnr, min_psnr, max_psnr):
    def pchipend(h1, h2, delta1, delta2):
        d = ((2 * h1 + h2) * delta1 - h1 * delta2) / (h1 + h2)
        if (d * delta1) < 0:
            d = 0
        elif (delta1 * delta2 < 0) and (abs(d) > abs(3 * delta1)):
            d = 3 * delta1
        return d

    log_rate = np.log10(rate[::-1])
    log_dist = psnr[::-1]

    H = log_dist[1:] - log_dist[:-1]
    delta = (log_rate[1:] - log_rate[:-1]) / H

    d = np.zeros(4)
    d[0] = pchipend(H[0], H[1], delta[0], delta[1])
    for i in [1, 2]:
        d[i] = (3 * H[i - 1] + 3 * H[i]) / ((2 * H[i] + H[i - 1]) / delta[i - 1] + (H[i] + 2 * H[i - 1]) / delta[i])
    d[3] = pchipend(H[2], H[1], delta[2], delta[1])

    c = np.zeros(3)
    b = np.zeros(3)
    for i in range(3):
        c[i] = (3 * delta[i] - 2 * d[i] - d[i + 1]) / H[i]
        b[i] = (d[i] - 2 * delta[i] + d[i + 1]) / (H[i] * H[i])

    # cubic function is rate(i) + s*(d(i) + s*(c(i) + s*(b(i))) where s = x - dist(i)
    # or rate(i) + s*d(i) + s*s*c(i) + s*s*s*b(i)
    # primitive is s*rate(i) + s*s*d(i)/2 + s*s*s*c(i)/3 + s*s*s*s*b(i)/4

    result = 0.0
    for i in range(3):
        s0 = log_dist[i]
        s1 = log_dist[i + 1]

        # clip s0 and s1 to valid range
        s0 = min(max(s0, min_psnr), max_psnr)
        s1 = min(max(s1, min_psnr), max_psnr)

        s0 = s0 - log_dist[i]
        s1 = s1 - log_dist[i]

        if s1 > s0:
            result = result + (s1 - s0) * log_rate[i]
            result = result + (s1 * s1 - s0 * s0) * d[i] / 2
            result = result + (s1 * s1 * s1 - s0 * s0 * s0) * c[i] / 3
            result = result + (s1 * s1 * s1 * s1 - s0 * s0 * s0 * s0) * b[i] / 4

    return result

