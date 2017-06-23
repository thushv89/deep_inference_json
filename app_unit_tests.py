from query_service import app,download_image,get_parameter
import unittest
import json
from flask import request
import os
import numpy as np

class FlaskServiceTests(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def tearDown(self):
        pass

    def test_assert_home_page_visit_success(self):
        result = self.app.get('/')
        self.assertEqual(result.status_code,200)

    def test_assert_get_prediction_via_url_success(self):
        with app.test_client() as c:
            result = self.app.get('/infer?filename=input_url_test.json&batch_size=1')
            string_data = result.data.decode("utf-8")

        self.assertIn("Egyptian cat",string_data) and self.assertIn("balloon",string_data)

    def test_assert_get_prediction_via_json_success(self):
        with app.test_client() as c:
            result = c.get('/infer_from_url?url=https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Kittyply_edit1.jpg/220px-Kittyply_edit1.jpg')
            string_data  =result.data.decode("utf-8")

        self.assertIn("Egyptian cat",string_data)

    def test_assert_downloaded_images_are_persisted(self):

        download_image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Kittyply_edit1.jpg/220px-Kittyply_edit1.jpg",'test-cat')

        file_exists = os.path.exists('test-cat.jpg')
        self.assertTrue(file_exists)

    def test_assert_provide_proper_error_for_wrong_extension_of_image(self):
        info_dict = download_image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Kittyply_edit1.jpg/220px-Kittyply_edit1",
            'test-cat')
        self.assertEqual(info_dict['error'],'Please provide urls of images only of JPG(JPEG) or PNG format (https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Kittyply_edit1.jpg/220px-Kittyply_edit1)')

    def test_assert_wrong_url_return_propper_error_message(self):
        info_dict = download_image(
            "https://upload.wikimedia/wikipedia/commons/thumb/b/bb/Kittyply_edit1.jpg/220px-Kittyply_edit1.jpg",
            'test-cat')
        self.assertIn('A URL error occured',info_dict['error'])

    def test_assert_convert_png_images_to_jpg(self):

        download_image("http://www.freepngimg.com/download/dog/8-dog-png-image-picture-download-dogs.png",'test-dog')

        file_exists = os.path.exists('test-dog.jpg')

        self.assertTrue(file_exists)

    def test_assert_vgg_weights_is_nonzero(self):
        any_nonzero = True
        TF_SCOPES = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
                     'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3',
                     'fc6', 'fc7', 'fc8']
        for scope in TF_SCOPES:
            any_nonzero = np.any(get_parameter(scope,'weights'))
            if not any_nonzero:
                break

        self.assertTrue(any_nonzero)

    def test_assert_vgg_bias_is_nonzero(self):
        any_nonzero = True
        TF_SCOPES = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
                     'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3',
                     'fc6', 'fc7', 'fc8']
        for scope in TF_SCOPES:
            any_nonzero = np.any(get_parameter(scope, 'bias'))
            if not any_nonzero:
                break

        self.assertTrue(any_nonzero)

if __name__ == '__main__':
    unittest.main()