from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()
absolute_image_paths = response.download({"keywords": "happy", "limit": 5, "format": "png", "chromedriver": "/home/matej/Desktop/chromedriver", "print_urls": True, "download": True})
