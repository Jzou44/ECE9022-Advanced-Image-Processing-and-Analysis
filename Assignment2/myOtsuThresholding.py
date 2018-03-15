import SimpleITK as sitk
import numpy as np


class myOtsuThresholding:
    def Otsu_threshold_from_Gray(self, Gray_image):  # The sole input to this class is the grayscale image
        Gray_image_Vector = sitk.GetArrayFromImage(Gray_image)
        flatten_Gray_image_Vector = Gray_image_Vector.flatten()
        total_pixel_number = len(flatten_Gray_image_Vector)
        best_otsu_value = np.inf
        best_threshold = 0
        for threshold in range(1, 256, 1):
            left_Vector = [x for x in flatten_Gray_image_Vector if x < threshold]
            right_Vector = [x for x in flatten_Gray_image_Vector if x >= threshold]
            if (len(left_Vector) - 1) > 0 and (len(right_Vector) - 1) > 0:
                left_w = len(left_Vector) / total_pixel_number
                # want unbiased estimator (n-1)
                left_variance = self.calc_variance(left_Vector)
                right_w = len(right_Vector) / total_pixel_number
                right_variance = self.calc_variance(right_Vector)
                otsu_value = left_w * left_variance + right_w * right_variance
                if otsu_value < best_otsu_value:
                    best_otsu_value = otsu_value
                    best_threshold = threshold
        # best_threshold=168
        print("Computed Otsu's threshold value={0}".format(best_threshold))
        # any pixel in the input grayscale image with intensity higher than or equal to the computed threshold value is converted to 1 (foreground)
        binary_image_Vector = (Gray_image_Vector > best_threshold).astype(np.int)*255
        binary_image = sitk.GetImageFromArray(binary_image_Vector)
        # The sole output of this class is another 2D SimpleITK::image object, with pixel type of sitkUInt8
        binary_image = sitk.Cast(binary_image, sitk.sitkUInt8)
        binary_image.CopyInformation(Gray_image)
        # The sole output of this class is a 2D, single channel, SimpleITK::Image object
        return binary_image

    def calc_variance(self, array):
        mean = np.sum(array) / len(array)
        variance = np.sum((array - mean) * (array - mean)) / (len(array)-1)
        return variance
