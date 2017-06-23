from query_service import app,reset_tf_graph
import unittest
import json
from flask import request
class FlaskServiceTests(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def tearDown(self):
        pass

    def test_assert_home_page_visit_success(self):
        result = self.app.get('/')
        self.assertEqual(result.status_code,200)

    def test_assert_infer_page_visit_success(self):
        with app.test_client() as c:
            result = self.app.get('/infer?filename=input_url_test.json&batch_size=1')
            string_data = result.data.decode("utf-8")

        self.assertIn("Egyptian cat",string_data) and self.assertIn("balloon",string_data)

    def test_assert_get_prediction_via_web_service(self):
        with app.test_client() as c:
            result = c.get('/infer_from_url?url=https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Kittyply_edit1.jpg/220px-Kittyply_edit1.jpg')
            string_data  =result.data.decode("utf-8")

        self.assertIn("Egyptian cat",string_data)

    #def assert_download_weights_file_if_not_exist():

    #def assert_if_weight_file_locally_exists():


    #def assert_downloaded_images_are_persisted():

    #def assert_convert_png_images_to_jpg():

    #def assert_vgg_weights_is_nonzero():

    #def assert_vgg_bias_is_nonzero():

    #def assert_prediction_class():

    #def assert_batch_size_is_not_a_factor_of_input_image_count()


if __name__ == '__main__':
    unittest.main()