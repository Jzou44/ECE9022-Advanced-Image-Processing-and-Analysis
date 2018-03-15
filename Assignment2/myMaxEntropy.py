import SimpleITK as sitk
import numpy as np


class myMaxEntropy:
    def max_entropy_threshold(self, input_image):# The sole input to this class is the grayscale image
        input_image_Vector = sitk.GetArrayFromImage(input_image)
        histogram = np.histogram(input_image_Vector, bins=256, range=[0, 256])
        PMF = 1.0 / np.sum(histogram[0]) * histogram[0]

        low_entropy = np.zeros(256)
        high_entropy = np.zeros(256)
        total_entropy = np.zeros(256)

        for threshold in range(0, 256, 1):
            low_PMF = PMF[:threshold]
            high_PMF = PMF[threshold:]

            if np.sum(low_PMF) > 0:
                low_non_zero_normalized_PMF = low_PMF[np.nonzero(low_PMF)] / np.sum(low_PMF)
                # calculate entropy
                low_entropy[threshold] = -np.sum(low_non_zero_normalized_PMF * np.log2(low_non_zero_normalized_PMF))

            if np.sum(high_PMF) > 0:
                high_non_zero_normalized_PMF = high_PMF[np.nonzero(high_PMF)] / np.sum(high_PMF)
                high_entropy[threshold] = -np.sum(high_non_zero_normalized_PMF * np.log2(high_non_zero_normalized_PMF))

            total_entropy[threshold] = low_entropy[threshold] + high_entropy[threshold]

        maxEntropyThreshold = np.argmax(total_entropy)
        print('Maximum entropy occurs at intensity:{0}'.format(maxEntropyThreshold))
        # any pixel in the input grayscale image with intensity higher than or equal to the computed threshold value is converted to 1 (foreground)
        binary_image_Vector = (input_image_Vector > maxEntropyThreshold).astype(np.int)*255
        binary_image = sitk.GetImageFromArray(binary_image_Vector)
        binary_image = sitk.Cast(binary_image, sitk.sitkUInt8)
        binary_image.CopyInformation(input_image)
        # The sole output of this class is a 2D, single channel, SimpleITK::Image object
        return binary_image
