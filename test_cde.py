"""Unit tests for cde.py"""

import unittest
import os
from cde import extract_figure_data, write_extracted_figure_data
from PIL import Image

class TestCde(unittest.TestCase):
    def test_extract_figure_data(self):
        """Test extract_figure_data function"""
        objects, clusters = extract_figure_data('test_images', 'test_output')
        # check that output directory exists, if not create it
        if not os.path.exists('test_output'):
            os.makedirs('test_output')
        print(objects)
        # write extracted figure data to JSON files and save images
        write_extracted_figure_data(objects, clusters, 'test_output')

if __name__ == '__main__':
    unittest.main()

