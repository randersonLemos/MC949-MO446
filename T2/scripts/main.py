import cv2
from classes.plot import Plot
from classes.imagemisc import  ImageMisc
from classes.featureDetector import FeatureDetector
from classes.superimage import SuperImage
from classes.superimagepair import SuperImagePair
from classes.camera import SimplePinholeCamera

ROOT_DIR_IMAGES = '../SampleSet/MVS Data/scan6_2_1'

if __name__ == '__main__':
    paths = ImageMisc.get_paths(ROOT_DIR_IMAGES, '*max.png')
    imags_color = ImageMisc.load_images(paths)
    # Plot.plot_images_grid(imags_color, 1, 2, (15, 10))

    imags_gray = list(map(ImageMisc.to_grayscale, imags_color))
    # Plot.plot_images_grid(imags_gray, 1, 2, (15, 10))

    FD = FeatureDetector(kind='sift')
    SUPERIMAGES = []
    for imag_color, imag_gray in zip(imags_color, imags_gray):
        keypoints, descriptors = FD.detect(imag_gray)
        si = SuperImage()
        si.set_imag_color(imag_color)
        si.set_imag_gray(imag_gray)
        si.set_keypoints(keypoints)
        si.set_descriptors(descriptors)
        SUPERIMAGES.append(si)

    imags_w_keypoints = [si.imag_w_keypoints() for si in SUPERIMAGES]
    # Plot.plot_images_grid(imags_w_keypoints, 1, 2, (15, 10))

    si1 = SUPERIMAGES[0]
    si2 = SUPERIMAGES[1]
    sip = SuperImagePair(si1, si2)

    sip.set_matcher(
        cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    )
    sip.match()

    imag_w_matches = sip.imag_w_matches(kind='good-matches')
    # Plot.plot_images_grid([imag_w_matches], 1, 1, (15, 10))

    sip.estimate_fundamental_matrix(ransacReprojThreshold=0.25, confidence=0.99)
    sip.evaluate_fundamental_estimation_quality()

    imag1_epipolar, imag2_epipolar = sip.imag_w_epipolar(num_points=200, scale=2.5, point_size=20)
    # Plot.plot_images_grid([imag1_epipolar, imag2_epipolar], 1, 2, (15, 10))

    imag_gray = next(iter(imags_gray))
    imag_H, imag_W = imag_gray.shape
    came_f = max(imag_H, imag_W)
    came_cx = imag_W/2
    came_cy = imag_H/2
    camera = SimplePinholeCamera(f=came_f, cx=came_cx, cy=came_cy)

    sip.set_intrinsic((camera.K()))

    sip.estimate_essential_matrix()
    sip.evaluate_essential_estimation_quality()

    sip.estimate_pose()
    sip.estimate_points3d()

    R1, t1 = sip.get_camera_1_pose()
    R2, t2 = sip.get_camera_2_pose()
    points3d = sip.get_points3d()
    Plot.plot_cameras_frustum(R1, t1, R2.T, t2, points3d)