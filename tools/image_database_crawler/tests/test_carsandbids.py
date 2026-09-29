from image_db_crawler.carsandbids import normalize_media_url, parse_visible_metadata


def test_normalize_media_url_removes_cloudflare_transform():
    url = "https://media.carsandbids.com/cdn-cgi/image/width%3D542%2Cquality%3D70/hash/photos/a.jpg?t=1"
    assert normalize_media_url(url) == "https://media.carsandbids.com/hash/photos/a.jpg?t=1"


def test_parse_visible_metadata():
    text = """2014 Falcon F7
Make
Falcon
Model
F7
Engine
7.0L V8
"""
    result = parse_visible_metadata(text)
    assert (result.year, result.make, result.model, result.engine) == ("2014", "Falcon", "F7", "7.0L V8")

