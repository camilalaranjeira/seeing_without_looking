import cv2, math 
import numpy as np
import imutils, dlib
from imutils import face_utils
from skimage import io, color
from matplotlib.path import Path
from scipy.signal import find_peaks, convolve2d


predictor = dlib.shape_predictor('fitzpatrick/shape_predictor_68_face_landmarks.dat')

class Segmentation():
    ##### Soure from: https://github.com/Jeanvit/PySkinDetection #####
    
    def run(self, image):
        
#         image = cv2.imread(filename)
        HSV_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        YCbCr_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
#         RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        binary_mask_image = self.color_segmentation(HSV_image, YCbCr_image)
        mask, output = self.region_based_segmentation(image, binary_mask_image)
        
        return image, mask, output
        
    
    def color_segmentation(self, HSV_image, YCbCr_image):

        lower_HSV_values = np.array([0, 40, 0], dtype = "uint8")
        upper_HSV_values = np.array([25, 255, 255], dtype = "uint8")

        lower_YCbCr_values = np.array((0, 138, 67), dtype = "uint8")
        upper_YCbCr_values = np.array((255, 173, 133), dtype = "uint8")

        # A binary mask is returned. White pixels (255) represent pixels that fall into the upper/lower.
        mask_YCbCr = cv2.inRange(YCbCr_image, lower_YCbCr_values, upper_YCbCr_values)
        mask_HSV = cv2.inRange(HSV_image, lower_HSV_values, upper_HSV_values) 

        binary_mask_image = cv2.add(mask_HSV,mask_YCbCr)
        return binary_mask_image
    
    
    def region_based_segmentation(self, image, binary_mask_image):
        image_foreground = cv2.erode(binary_mask_image, None, iterations = 3) # Remove noise
        dilated_binary_image = cv2.dilate(binary_mask_image, None, iterations = 3) # Reduce background 
        ret, image_background = cv2.threshold(dilated_binary_image, 1, 128, cv2.THRESH_BINARY)  #set background to 128
        
        image_marker = np.int32(cv2.add(image_foreground, image_background))  # watershed markers
        cv2.watershed(image, image_marker)
        mask = cv2.convertScaleAbs(image_marker) # convert back to uint8 
        
        # bitwise of the mask with the input image
        ret, image_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        output = cv2.bitwise_and(image, image, mask=image_mask)
        
        return image_mask, output
    

class SkinTone():
    ##### Source from https://github.com/mattgroh/fitzpatrick17k/blob/main/ita_fitzpatrick_analysis.ipynb
    
    
    def get_slices(self, smooth, shape):

        h, w = smooth.shape
        y, x = np.mgrid[:h, :w]
        points = np.transpose((x.ravel(), y.ravel()))

        #left cheek
        lcheek = []
        lcheek.extend(shape[:6].tolist())
        lcheek.append(shape[31])

        path = Path( lcheek )
        mask = path.contains_points( points )
        mask = mask.reshape(h, w)

        lcheek = np.full(smooth.shape, np.nan)
        lcheek[mask] = smooth[mask]

        #right cheek
        rcheek = []
        rcheek.extend(shape[11:17].tolist())
        rcheek.append(shape[35])

        path = Path( rcheek )
        mask = path.contains_points( points )
        mask = mask.reshape(h, w)

        rcheek = np.full(smooth.shape, np.nan)
        rcheek[mask] = smooth[mask]

        #chin
        chin = []
        chin.extend(shape[6:11].tolist())
        chin.append(shape[57])

        path = Path( chin )
        mask = path.contains_points( points )
        mask = mask.reshape(h, w)

        chin = np.full(smooth.shape, np.nan)
        chin[mask] = smooth[mask]

        return lcheek, rcheek, chin
    
    
    def ITA(self, img):
        
        shape = predictor(img, dlib.rectangle(0, 0, img.shape[1], img.shape[0]))
        shape = imutils.face_utils.shape_to_np(shape)

        # ITA MAP
        ita_img, l, b, a = self.get_ita_map(img)
    #     plt.figure()
    #     plt.imshow(ita_img, cmap='coolwarm', vmin=-100, vmax=120)

        # AVERAGE FILTER
        avg = np.full((3,3), 1/9)
        smooth_ = convolve2d(ita_img, avg, mode='same')    

        # SKIN SEGMENTATION
        img, mask, output = Segmentation().run(img)
        imgshape = smooth_.shape
        s = smooth_.flatten()
        s[~mask.astype(bool).flatten()] = np.nan
        smooth = s.reshape(imgshape)

        lcheek, rcheek, chin = self.get_slices(smooth, shape)
    #     plt.figure()
    #     plt.imshow(lcheek, cmap='coolwarm', vmin=-100, vmax=120)
    #     plt.imshow(rcheek, cmap='coolwarm', vmin=-100, vmax=120)
    #     plt.imshow(chin, cmap='coolwarm', vmin=-100, vmax=120)

        italst, regionlst = [], []
        for slc in [lcheek, rcheek, chin]:
            f = slc.flatten()
            f = f[~np.isnan(f)]
            if len(f) == 0: continue

            values, interv = np.histogram(f, bins=50)
            window = 5
            mav = np.convolve(values, np.ones(window), 'same') / window

            peaks, props = find_peaks(mav, mav, prominence=1)
            if len(peaks) == 0: 
                ita = np.mean(f)
                italst.append(ita)
            else:
                maxheight_idx = np.argmax(props['peak_heights'])
                ita = interv[peaks[maxheight_idx]]
                italst.append(ita)

    #         plt.figure()
    #         plt.bar(interv[:-1], values, width=1)
    #         plt.plot(interv[:-1],mav)
    #         plt.axvline(ita, 0, 1, color='r')
    #         plt.show()
            
        px_idx = np.argmax(ita_img.flatten() - np.mean(italst))
        row = max(min(px_idx % ita_img.shape[0], ita_img.shape[0]-3), 3)  
        col = max(min(px_idx // ita_img.shape[0], ita_img.shape[1]-3), 3)
        region = img[row-3:row+3, col-3:col+3]
        
        return np.mean(italst), region
    
    def get_ita_map(self, image):
        """
        Calculates the individual typology angle (ITA) for a given 
        RGB image.
        """

        # Convert to CIE-LAB color space
#         RGB = Image.open(image)
        CIELAB = np.array(color.rgb2lab(image))

        # Get L and B (subset to +- 1 std from mean)
        L = CIELAB[:, :, 0]
        A = CIELAB[:, :, 1]
        B = CIELAB[:, :, 2]
        ITA = np.arctan2(L-50, B) * (180/np.pi)
         
#         L = np.where(L != 0, L, np.nan)
#         std, mean = np.nanstd(L), np.nanmean(L)
#         L = np.where(L >= mean - std, L, np.nan)
#         L = np.where(L <= mean + std, L, np.nan)
#         Lmask = np.where(np.isnan(L), 0, 255) 
        
        
#         B = np.where(B != 0, B, np.nan)
#         std, mean = np.nanstd(B), np.nanmean(B)
#         B = np.where(B >= mean - std, B, np.nan)
#         B = np.where(B <= mean + std, B, np.nan)
#         Bmask = np.where(np.isnan(L), 0, 255) 

        return ITA, L, B, A
    
    
    
#         def ITA(self, image):
#         """
#         Calculates the individual typology angle (ITA) for a given 
#         RGB image.
#         """

#         # Convert to CIE-LAB color space
# #         RGB = Image.open(image)
#         CIELAB = np.array(color.rgb2lab(image))

#         # Get L and B (subset to +- 1 std from mean)
#         L = CIELAB[:, :, 0]
#         L = np.where(L != 0, L, np.nan)
#         std, mean = np.nanstd(L), np.nanmean(L)
#         L = np.where(L >= mean - std, L, np.nan)
#         L = np.where(L <= mean + std, L, np.nan)

#         B = CIELAB[:, :, 2]
#         B = np.where(B != 0, B, np.nan)
#         std, mean = np.nanstd(B), np.nanmean(B)
#         B = np.where(B >= mean - std, B, np.nan)
#         B = np.where(B <= mean + std, B, np.nan)

#         # Calculate ITA
        
        
#         ITA = math.atan2(np.nanmean(L) - 50, np.nanmean(B)) * (180 / np.pi)

#         return ITA
    
    
    
    
    def ita2str(self, ITA, method='kinyanjui'):
                
        if method == 'kinyanjui':
            if ITA > 55:
                return ('very light', 1)
            elif ITA > 41:
                return ('light 2',  2)
            elif ITA > 28:
                return ('light', 3)
            elif ITA > 19:
                return ('intermediate 2', 4)
            elif ITA > 10:
                return ('intermediate', 5)
            elif ITA <= 10:
                return ('dark', 6)
            else:
                return None

        # Use empirical thresholds
        else: #Fitzpatrick17k
            if ITA >= 45:
                return ('very light', 1)
            elif ITA > 28:
                return ('light 2',  2)
            elif ITA > 17:
                return ('light', 3)
            elif ITA > 5:
                return ('intermediate 2', 4)
            elif ITA > -20:
                return ('intermediate', 5)
            elif ITA <= -20:
                return ('dark', 6)
            else:
                return None

    
    
    
    
    
    
    