import SimpleITK as sitk


class RGB2Gray:
    def convert_RBG_to_Gray(self, RGB_image_file="inputImg_RGB.jpg"):
        #read the file using SimpleITK class
        RGB_image = sitk.ReadImage(RGB_image_file)
        #convert the input image to a numpy array
        RGB_image_Vector = sitk.GetArrayFromImage(RGB_image)
        #The use of a weighting scheme for converting RGB to grayscale image
        Gray_image_Vector = 0.299 * RGB_image_Vector[:, :, 0] + 0.587 * RGB_image_Vector[:, :, 1] + 0.114 * RGB_image_Vector[:, :, 2]
        Gray_image = sitk.GetImageFromArray(Gray_image_Vector)
        Gray_image = sitk.Cast(Gray_image, sitk.sitkUInt8)
        Gray_image.CopyInformation(RGB_image)
        #The sole output of this class is a 2D, single channel, SimpleITK::Image object
        return Gray_image


