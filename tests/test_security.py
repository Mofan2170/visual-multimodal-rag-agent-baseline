import unittest

import httpx
from fastapi.responses import Response

from app.main import _is_same_origin, _secure_response, app


class SecurityTests(unittest.IsolatedAsyncioTestCase):
    def test_same_origin_check_rejects_cross_site_requests(self) -> None:
        self.assertTrue(_is_same_origin("http://127.0.0.1:8000", "127.0.0.1:8000"))
        self.assertFalse(_is_same_origin("https://example.com", "127.0.0.1:8000"))
        self.assertFalse(_is_same_origin("null", "127.0.0.1:8000"))

    def test_api_response_receives_security_headers(self) -> None:
        response = _secure_response(Response(), "/api/runtime/status")

        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])

    async def test_cross_origin_runtime_change_is_rejected(self) -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/api/runtime/config",
                headers={"Origin": "https://example.com"},
                json={"chat_model": "should-not-be-applied"},
            )

        self.assertEqual(response.status_code, 403)

    async def test_invalid_base_url_returns_bad_request(self) -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/api/runtime/config",
                headers={"Origin": "http://testserver"},
                json={"base_url": "file:///tmp/service"},
            )

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
