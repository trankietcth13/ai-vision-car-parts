import pytest

from image_db_crawler import cli


def test_network_crawl_requires_authorization_file(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["vehicle-image-crawler", "--source", "carsandbids", "--url", "https://carsandbids.com/auctions/x/car"],
    )
    with pytest.raises(SystemExit, match="Terms prohibit"):
        cli.main()
