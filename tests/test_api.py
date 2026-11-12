from fastapi.testclient import TestClient

from api.main import create_app


def make_client(settings):
    app = create_app(settings)
    return TestClient(app)


def test_health(settings):
    client = make_client(settings)
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_index_page_renders(settings):
    client = make_client(settings)
    res = client.get("/")
    assert res.status_code == 200
    assert "IntelliDoc" in res.text


def test_upload_extract_and_inspect(settings, fixture_pdf):
    path, labels = fixture_pdf
    client = make_client(settings)
    with path.open("rb") as fh:
        res = client.post("/api/documents", files={"file": ("invoice.pdf", fh, "application/pdf")})
    assert res.status_code == 200, res.text
    payload = res.json()
    doc_id = payload["doc_id"]
    assert payload["fields"]["total_amount"]["value"] == labels["total_amount"]

    detail = client.get(f"/api/documents/{doc_id}").json()
    assert detail["fields"]
    image = client.get(f"/api/documents/{doc_id}/page/0.png")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"

    listing = client.get("/api/documents").json()
    assert listing["stats"]["documents"] == 1


def test_correction_updates_field(settings, fixture_pdf):
    path, _ = fixture_pdf
    client = make_client(settings)
    with path.open("rb") as fh:
        doc_id = client.post("/api/documents", files={"file": ("invoice.pdf", fh, "application/pdf")}).json()["doc_id"]

    res = client.post(f"/api/documents/{doc_id}/correct", json={"field": "total_amount", "value": 11800.5, "reviewer": "qa"})
    assert res.status_code == 200
    fields = {f["name"]: f for f in res.json()["document"]["fields"]}
    assert fields["total_amount"]["corrected"] == 11800.5


def test_rejects_bad_field_and_bad_type(settings, fixture_pdf):
    path, _ = fixture_pdf
    client = make_client(settings)
    with path.open("rb") as fh:
        doc_id = client.post("/api/documents", files={"file": ("invoice.pdf", fh, "application/pdf")}).json()["doc_id"]
    assert client.post(f"/api/documents/{doc_id}/correct", json={"field": "nope", "value": 1}).status_code == 422
    res = client.post("/api/documents", files={"file": ("x.exe", b"binary", "application/octet-stream")})
    assert res.status_code == 415


def test_search_and_approve(settings, fixture_pdf):
    path, _ = fixture_pdf
    client = make_client(settings)
    with path.open("rb") as fh:
        doc_id = client.post("/api/documents", files={"file": ("invoice.pdf", fh, "application/pdf")}).json()["doc_id"]

    hits = client.get("/api/search", params={"q": "Nandini"}).json()["hits"]
    assert hits and hits[0]["page"] == 0

    approved = client.post(f"/api/documents/{doc_id}/approve")
    assert approved.status_code == 200
    assert approved.json()["document"]["status"] == "auto_approved"
