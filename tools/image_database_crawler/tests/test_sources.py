import httpx

from image_db_crawler.sources import WikimediaSource, infer_vehicle


def test_infer_vehicle_from_title():
    vehicle = infer_vehicle("2013-2017 Honda Accord Car Battery Replacement")
    assert vehicle.year == "2013-2017"
    assert vehicle.make == "Honda"
    assert vehicle.model == "Accord"

    repeated = infer_vehicle("2013-2017 Honda Accord", "2013-2017 Honda Accord Car Battery")
    assert repeated.model == "Accord"

    compact_tag = infer_vehicle("Summer project car", "1971 triumphspitfire")
    assert (compact_tag.year, compact_tag.make, compact_tag.model) == ("1971", "Triumph", "spitfire")


def test_wikimedia_keeps_license_and_attribution():
    payload = {
        "query": {
            "pages": [
                {
                    "title": "File:2015 Toyota Prius engine.jpg",
                    "imageinfo": [
                        {
                            "url": "https://upload.wikimedia.org/test.jpg",
                            "descriptionurl": "https://commons.wikimedia.org/wiki/File:test.jpg",
                            "extmetadata": {
                                "LicenseShortName": {"value": "CC BY-SA 4.0"},
                                "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0/"},
                                "Artist": {"value": "<b>Example Author</b>"},
                                "Credit": {"value": "Own work"},
                            },
                        }
                    ],
                }
            ]
        }
    }

    def handler(request):
        return httpx.Response(200, json=payload)

    source = WikimediaSource("test/1.0 (test@example.com)")
    source.client.close()
    source.client = httpx.Client(transport=httpx.MockTransport(handler))
    listing = source.collect(["Automobiles with open hoods"], limit=1)[0]
    source.close()
    image = listing.images[0]
    assert image.license_code == "cc-by-sa-4.0"
    assert image.creator == "Example Author"
    assert image.vehicle.year == "2015"
    assert image.vehicle.make == "Toyota"
