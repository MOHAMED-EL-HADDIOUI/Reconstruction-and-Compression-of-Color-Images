import numpy
from PIL import Image
import numpy as np
from scipy.linalg import svd


def openImage(imagePath):
    imOrig = Image.open(imagePath)
    im = numpy.array(imOrig)
    aRed = im[:, :, 0]
    aGreen = im[:, :, 1]
    aBlue = im[:, :, 2]
    return [aRed, aGreen, aBlue, imOrig]


def compressSingleChannel(channelDataMatrix, singularValuesLimit):
    uChannel, sChannel, vhChannel = numpy.linalg.svd(channelDataMatrix)
    aChannelCompressed = numpy.zeros(
        (channelDataMatrix.shape[0], channelDataMatrix.shape[1]))
    k = singularValuesLimit
    leftSide = numpy.matmul(uChannel[:, 0:k], numpy.diag(sChannel)[0:k, 0:k])
    aChannelCompressedInner = numpy.matmul(leftSide, vhChannel[0:k, :])
    aChannelCompressed = aChannelCompressedInner.astype('uint8')
    return aChannelCompressed


aRed, aGreen, aBlue, originalImage = openImage("dhoni.jpg")
for i in (20, 100, 600):
    aRedCompressed = compressSingleChannel(aRed, i)
    aGreenCompressed = compressSingleChannel(aGreen, i)
    aBlueCompressed = compressSingleChannel(aBlue, i)
    imr = Image.fromarray(aRedCompressed, mode=None)
    img = Image.fromarray(aGreenCompressed, mode=None)
    imb = Image.fromarray(aBlueCompressed, mode=None)
    newImage = Image.merge("RGB", (imr, img, imb))
    newImage.show()
