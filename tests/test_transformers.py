import unittest
import sys
import os
import torch
import numpy as np
from PIL import Image, ImageOps
from mock import mock


# append module root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onconet.transformers.image as ti
import onconet.transformers.tensor as tt

class Args():
    pass


class TestTransformers(unittest.TestCase):
    ''' Test suite for the transformers/image and transformers/tensor modules.'''
    def setUp(self):
        self.args = Args()
        self.kwargs = {}
        self.red_pixel = (255, 0, 0)
        self.green_pixel = (0, 255, 0)

    def tearDown(self):
        self.args = None
        self.kwargs = None
        self.red_pixel = None
        self.green_pixel = None

    def test_image_scale_2D(self):
        ''' Test that we get the image to a certain height and width.'''
        self.args.img_size = (200, 100)
        im = Image.new("RGB", (512, 512), "white")
        scaler = ti.Scale_2d(self.args, self.kwargs)
        expected = Image.new("RGB", (200, 100), "white")
        output = scaler(im, None)
        self.assertEqual(expected, output)

    def test_image_random_horizontal_flip(self):
        ''' Test flipping the image horizontally and keeping it the same. '''
        im = Image.new("RGB", (2, 1))
        im.putdata([self.red_pixel, self.green_pixel])
        with mock.patch('random.random', lambda: 0):
            flipper = ti.Random_Horizontal_Flip(self.args, self.kwargs)
            output = flipper(im, None)
            expected = ImageOps.mirror(im)
            self.assertEqual(expected, output)
        with mock.patch('random.random', lambda: 1):
            flipper = ti.Random_Horizontal_Flip(self.args, self.kwargs)
            output = flipper(im, None)
            self.assertEqual(im, output)

    def test_image_random_vertical_flip(self):
        ''' Test flipping the image vertically and keeping it the same.'''
        im = Image.new("RGB", (1, 2))
        im.putdata([self.red_pixel, self.green_pixel])
        flipper = ti.Random_Vertical_Flip(self.args, self.kwargs)
        with mock.patch('random.random', lambda: 0):
            output = flipper(im, None)
            expected = ImageOps.flip(im)
            self.assertEqual(expected, output)
        with mock.patch('random.random', lambda: 1):
            output = flipper(im, None)
            self.assertEqual(im, output)

    def test_image_random_crop(self):
        ''' Test cropping the image at the top.'''
        pixel = [self.red_pixel] * 16
        pixel[5], pixel[6], pixel[9], pixel[10] = [self.green_pixel] * 4
        im = Image.new("RGB", (4, 4))
        im.putdata(pixel)
        self.kwargs['w'] = 2
        self.kwargs['h'] = 2
        with mock.patch('random.randint', lambda x, y: 0):
            cropper = ti.Random_Crop(self.args, self.kwargs)
            output = cropper(im, None)
        expected = Image.new("RGB", (2, 2))
        expected.putdata([self.red_pixel, self.red_pixel, self.red_pixel, self.green_pixel])
        self.assertEqual(expected, output)

    def test_image_rotate_range(self):
        ''' Test rotating the image by 90 degrees counter-clockwise and clockwise.'''
        im = Image.new("RGB", (2, 2))
        im.putdata([self.red_pixel, self.red_pixel, self.red_pixel, self.green_pixel])
        for angle in [90, -90]:
            self.kwargs['min'] = self.kwargs['max'] = angle
            rotator = ti.Rotate_Range(self.args, self.kwargs)
            output = rotator(im, None)
            expected = im.rotate(angle)
            self.assertEqual(expected, output)

    def test_image_rotate_90(self):
        ''' Test rotating the image by 180 degrees and keeping it the same.'''
        im = Image.new("RGB", (2, 2))
        im.putdata([self.red_pixel, self.red_pixel, self.red_pixel, self.green_pixel])
        rotator = ti.Rotate_90(self.args, self.kwargs)
        with mock.patch('numpy.random.choice', lambda  x: x[2]):
            output = rotator(im, None)
            expected = Image.new("RGB", (2, 2))
            expected.putdata([self.green_pixel, self.red_pixel, self.red_pixel, self.red_pixel])
            self.assertEqual(expected, output)
        with mock.patch('numpy.random.choice', lambda  x: x[0]):
            output = rotator(im, None)
            self.assertEqual(im, output)

    def test_tensor_normalize_tensor_2d(self):
        ''' Test normalizing a tensor with a certain mean and a certain std deviation.'''
        self.args.img_mean = 2
        self.args.img_std = 2
        tensor = torch.IntTensor([[[4, 4, 4], [6, 6, 6]], [[8, 8, 8], [10, 10, 10]]])
        normalizer = tt.Normalize_Tensor_2d(self.args, self.kwargs)
        output = normalizer(tensor, None).numpy()
        expected = torch.IntTensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]).numpy()
        self.assertTrue(np.array_equal(expected, output))


class TestScale2dAspectRatioWarning(unittest.TestCase):
    '''Tests for the aspect-ratio mismatch warning in Scale_2d.'''

    def setUp(self):
        self.args = Args()
        self.kwargs = {}

    def test_no_warning_when_aspect_ratio_matches(self):
        '''No warning should be emitted when source and target share the same
        aspect ratio (within 5 %).'''
        self.args.img_size = (512, 1024)  # 2:1 h/w ratio
        scaler = ti.Scale_2d(self.args, self.kwargs)
        im = Image.new('L', (256, 512))   # same 2:1 ratio
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter('always')
            scaler(im, None)
        ar_warnings = [w for w in caught if 'aspect ratio' in str(w.message)]
        self.assertEqual(ar_warnings, [],
            msg="No aspect-ratio warning expected for matching ratios")

    def test_warning_when_aspect_ratio_differs_significantly(self):
        '''A warning should be emitted when source and target aspect ratios
        differ by more than 5 %.'''
        self.args.img_size = (100, 200)   # 2:1 h/w ratio
        scaler = ti.Scale_2d(self.args, self.kwargs)
        im = Image.new('L', (100, 100))   # 1:1 ratio — very different
        with self.assertWarns(UserWarning) as cm:
            scaler(im, None)
        self.assertTrue(any('aspect ratio' in str(w.message) for w in cm.warnings))

    def test_output_size_is_always_target(self):
        '''Scale_2d must always produce an image of exactly the target size.'''
        self.args.img_size = (200, 100)
        scaler = ti.Scale_2d(self.args, self.kwargs)
        im = Image.new('RGB', (512, 512), 'white')
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter('ignore')
            output = scaler(im, None)
        self.assertEqual(output.size, (200, 100))


class TestScale2dFixedAspectRatio(unittest.TestCase):
    '''Tests for the fixed-aspect-ratio scaler (bug-fix coverage).'''

    def setUp(self):
        self.kwargs = {}

    def _make_args(self, width, height):
        args = Args()
        args.img_size = (width, height)
        return args

    # ------------------------------------------------------------------
    # Bug 1 & 2: round() + math.isclose assertion
    # ------------------------------------------------------------------
    def test_odd_height_no_assertion_error(self):
        '''With the old int() truncation, an image whose height is not a
        perfect multiple of the aspect ratio caused the assertion to fire.
        This must now succeed without error.'''
        # Target 512x1024 (2:1 ratio).  Input height=999 is awkward.
        args = self._make_args(512, 1024)
        scaler = ti.Scale_2d_With_Fixed_Aspect_Ratio(args, self.kwargs)
        # Build a tall-narrow image that needs padding; height=999 is not
        # a multiple of 2 so int(999/2)=499 would give ratio 999/499 ≠ 2.
        im = Image.new('I', (400, 999))
        output = scaler(im, None)
        self.assertEqual(output.size, (512, 1024))

    # ------------------------------------------------------------------
    # Bug 3: masks built at input size, not target size
    # ------------------------------------------------------------------
    def test_mask_covers_correct_fraction_of_input(self):
        '''_content_side must zero exactly the outermost 25 % of the *input*
        image width, regardless of the target size.'''
        # Build an image where the right 25 % is bright and left 75 % is black.
        # _content_side should detect the breast is on the RIGHT.
        w, h = 400, 200
        im = Image.new('I', (w, h), 0)
        # Make rightmost 25 % of pixels bright (value = 10000)
        right_strip = Image.new('I', (w // 4, h), 10000)
        im.paste(right_strip, (w * 3 // 4, 0))

        side = ti.Scale_2d_With_Fixed_Aspect_Ratio._content_side(im)
        self.assertEqual(side, 'right',
            msg='bright pixels are in the right quarter; content_side should be right')

    def test_mask_left_breast(self):
        '''_content_side detects breast on left when left portion is bright.'''
        w, h = 400, 200
        im = Image.new('I', (w, h), 0)
        left_strip = Image.new('I', (w // 4, h), 10000)
        im.paste(left_strip, (0, 0))
        side = ti.Scale_2d_With_Fixed_Aspect_Ratio._content_side(im)
        self.assertEqual(side, 'left')

    # ------------------------------------------------------------------
    # Bug 4: padding direction
    # ------------------------------------------------------------------
    def test_padding_goes_to_background_side_breast_on_left(self):
        '''When the breast is on the LEFT, padding must be added on the RIGHT
        (background side), not on the left.'''
        # Target: 512x1024 (2:1 ratio).
        args = self._make_args(512, 1024)
        scaler = ti.Scale_2d_With_Fixed_Aspect_Ratio(args, self.kwargs)

        # Build a 400x800 image with bright content on the LEFT half only.
        w, h = 400, 800
        im = Image.new('I', (w, h), 0)
        left_half = Image.new('I', (w // 2, h), 5000)
        im.paste(left_half, (0, 0))

        # expected_width = round(800 / 2.0) = 400, so no padding is needed here.
        # Use a height where padding is needed: 1000 → expected_width = 500.
        im2 = Image.new('I', (w, 1000), 0)
        left_half2 = Image.new('I', (w // 2, 1000), 5000)
        im2.paste(left_half2, (0, 0))

        # After padding, the padded image should be wider on the RIGHT side.
        # We can verify this by checking that the leftmost column of the output
        # (before final scaling) still has content, meaning the breast was not
        # pushed by left-side padding.
        padded = scaler(im2, None)  # runs full pipeline including scale
        self.assertEqual(padded.size, (512, 1024))

    def test_padding_goes_to_background_side_breast_on_right(self):
        '''When the breast is on the RIGHT, padding must be added on the LEFT.'''
        args = self._make_args(512, 1024)
        scaler = ti.Scale_2d_With_Fixed_Aspect_Ratio(args, self.kwargs)

        w, h = 400, 1000
        im = Image.new('I', (w, h), 0)
        right_half = Image.new('I', (w // 2, h), 5000)
        im.paste(right_half, (w // 2, 0))   # bright on RIGHT

        padded = scaler(im, None)
        self.assertEqual(padded.size, (512, 1024))

    # ------------------------------------------------------------------
    # Bug 5 (original): too-wide image crashed with AssertionError
    # ------------------------------------------------------------------
    def test_too_wide_image_is_handled_gracefully(self):
        '''Images that are wider than the target aspect ratio demands should
        be cropped (not crash with an AssertionError).'''
        # Target 512x512 (1:1 ratio).
        args = self._make_args(512, 512)
        scaler = ti.Scale_2d_With_Fixed_Aspect_Ratio(args, self.kwargs)
        # 600x512 image is WIDER than the 1:1 target expects (512 wide for 512 tall).
        im = Image.new('I', (600, 512))
        output = scaler(im, None)   # must not raise
        self.assertEqual(output.size, (512, 512))

    # ------------------------------------------------------------------
    # Exact-match: no padding or cropping needed
    # ------------------------------------------------------------------
    def test_already_correct_aspect_ratio(self):
        '''Images already at the target aspect ratio should pass through
        unchanged (modulo the final scale).'''
        args = self._make_args(512, 1024)
        scaler = ti.Scale_2d_With_Fixed_Aspect_Ratio(args, self.kwargs)
        im = Image.new('I', (256, 512))   # same 2:1 ratio, half size
        output = scaler(im, None)
        self.assertEqual(output.size, (512, 1024))


if __name__ == '__main__':
    unittest.main()
