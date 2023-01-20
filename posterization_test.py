"""Unit tests for posterization_new.py"""
import unittest
import os
import numpy as np

from posterization import classify_pixels

COLORS= [
    (0.99, 0.99, 0.99), # white
    (0.99, 0.01, 0.01), # red
    (0.01, 0.99, 0.01), # green
    (0.01, 0.01, 0.99), # blue
    (0.99, 0.99, 0.01), # yellow
    (0.99, 0.01, 0.99), # magenta
    (0.01, 0.99, 0.99), # cyan
    (0.01, 0.01, 0.01), # black
    (0.99, 0.5, 0.01), # orange
    (0.5, 0.99, 0.01), # lime
    (0.01, 0.99, 0.5), # spring green
    (0.01, 0.5, 0.99), # azure
    (0.5, 0.01, 0.99), # violet
    (0.99, 0.01, 0.5), # rose
    (0.5, 0.5, 0.5), # gray
    (0.75, 0.75, 0.75), # light gray
    (0.25, 0.25, 0.25), # dark gray
    (0.5, 0.5, 0.01), # olive
    (0.5, 0.01, 0.5), # purple
    (0.01, 0.5, 0.5), # teal
    (0.5, 0.01, 0.01), # maroon
    ]

class TestPosterization(unittest.TestCase):

    def setUp(self):
        # creates 10x10 images with pixel values chose from n colors
        self.images = []
        for n in range(1, 11):
            image = np.random.randint(0, n, (10, 10))
            colors_shuffled = np.random.permutation(COLORS)
            image = np.array([colors_shuffled[i] for i in image])
            self.images.append(image)

    def test_classify_pixels(self):
        """Tests that classify_pixels returns a numpy array of the same shape as image."""
        for i, image in enumerate(self.images):
            classified, palette = classify_pixels(image)
            self.assertEqual(image.shape, classified.shape)
            self.assertEqual(len(palette),i+1)
            print(f"Testing image with {i+1} colors: found {len(palette)} colors")
            self.assertEqual(len(np.unique(classified)), i+1)
    
if __name__ == '__main__':
    unittest.main()