from __future__ import annotations

import base64
import hashlib
import json
import queue
import stat
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request

import siwc_spike
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt


class AuthorizationRequestTests(unittest.TestCase):
    def test_default_state_paths_are_anchored_to_the_spike_directory(self) -> None:
        parser = siwc_spike.build_parser()

        login = parser.parse_args(["login"])
        probe = parser.parse_args(["probe-saved"])

        self.assertEqual(login.host_id_file, siwc_spike.SPIKE_STATE_DIR / "host-id")
        self.assertEqual(login.output, siwc_spike.SPIKE_STATE_DIR / "credentials.json")
        self.assertEqual(probe.credentials, siwc_spike.SPIKE_STATE_DIR / "credentials.json")

    def test_initial_authorization_uses_dynamic_client_pkce_and_loopback(self) -> None:
        pending = siwc_spike.create_authorization(
            host_id="urn:uuid:11111111-1111-4111-8111-111111111111",
            port=1455,
            state="state-value",
            nonce="nonce-value",
            verifier="v" * 64,
        )

        parsed = urlparse(pending.authorization_url)
        params = parse_qs(parsed.query)
        expected_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(("v" * 64).encode()).digest()
        ).rstrip(b"=").decode()

        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.netloc, "auth.openai.com")
        self.assertEqual(parsed.path, "/api/accounts/authorize")
        self.assertEqual(params["client_id"], ["dynamic_agent_client"])
        self.assertEqual(params["agent_name_hint"], ["Securo"])
        self.assertEqual(params["redirect_uri"], ["http://127.0.0.1:1455/auth/callback"])
        self.assertEqual(params["scope"], [siwc_spike.REQUIRED_SCOPE_STRING])
        self.assertEqual(params["resource"], ["https://api.openai.com/v1"])
        self.assertEqual(params["state"], ["state-value"])
        self.assertEqual(params["nonce"], ["nonce-value"])
        self.assertEqual(params["code_challenge_method"], ["S256"])
        self.assertEqual(params["code_challenge"], [expected_challenge])

    def test_callback_rejects_wrong_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "state mismatch"):
            siwc_spike.validate_callback(
                {"code": ["abc"], "state": ["wrong"], "client_id": ["oaiapp_123"]},
                expected_state="expected",
                returning_client_id=None,
            )

    def test_new_registration_requires_issued_client_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "issued client_id"):
            siwc_spike.validate_callback(
                {"code": ["abc"], "state": ["expected"]},
                expected_state="expected",
                returning_client_id=None,
            )

    def test_returning_registration_rejects_changed_client_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "different client_id"):
            siwc_spike.validate_callback(
                {
                    "code": ["abc"],
                    "state": ["expected"],
                    "client_id": ["oaiapp_other"],
                },
                expected_state="expected",
                returning_client_id="oaiapp_saved",
            )

    def test_invalid_callback_does_not_consume_attempt(self) -> None:
        results: queue.Queue[siwc_spike.CallbackOutcome] = queue.Queue(maxsize=1)

        accepted = siwc_spike.offer_callback(
            {"code": ["poison"], "state": ["wrong"], "client_id": ["oaiapp_bad"]},
            expected_state="expected",
            returning_client_id=None,
            result_queue=results,
        )

        self.assertFalse(accepted)
        self.assertTrue(results.empty())

    def test_valid_state_oauth_denial_terminates_attempt(self) -> None:
        results: queue.Queue[siwc_spike.CallbackOutcome] = queue.Queue(maxsize=1)

        accepted = siwc_spike.offer_callback(
            {
                "error": ["access_denied"],
                "error_description": ["The user denied access"],
                "state": ["expected"],
            },
            expected_state="expected",
            returning_client_id=None,
            result_queue=results,
        )

        self.assertTrue(accepted)
        outcome = results.get_nowait()
        self.assertIsInstance(outcome, siwc_spike.CallbackFailure)
        assert isinstance(outcome, siwc_spike.CallbackFailure)
        self.assertEqual(outcome.message, "OpenAI authorization failed: The user denied access")

    def test_duplicate_valid_callback_does_not_block(self) -> None:
        results: queue.Queue[siwc_spike.CallbackResult] = queue.Queue(maxsize=1)
        params = {
            "code": ["good"],
            "state": ["expected"],
            "client_id": ["oaiapp_good"],
        }

        self.assertTrue(
            siwc_spike.offer_callback(
                params,
                expected_state="expected",
                returning_client_id=None,
                result_queue=results,
            )
        )
        self.assertFalse(
            siwc_spike.offer_callback(
                params,
                expected_state="expected",
                returning_client_id=None,
                result_queue=results,
            )
        )


class TokenAndStreamTests(unittest.TestCase):
    def test_id_token_signature_audience_issuer_subject_and_nonce_are_validated(self) -> None:
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_numbers = private_key.public_key().public_numbers()
        def encode_int(value: int) -> str:
            return base64.urlsafe_b64encode(
                value.to_bytes((value.bit_length() + 7) // 8, "big")
            ).rstrip(b"=").decode()
        jwks = {
            "keys": [
                {
                    "kty": "RSA",
                    "kid": "test-key",
                    "use": "sig",
                    "alg": "RS256",
                    "n": encode_int(public_numbers.n),
                    "e": encode_int(public_numbers.e),
                }
            ]
        }
        token = jwt.encode(
            {
                "iss": siwc_spike.ISSUER,
                "aud": "oaiapp_test",
                "sub": "subject-123",
                "nonce": "nonce-123",
                "exp": int(time.time()) + 300,
            },
            private_key,
            algorithm="RS256",
            headers={"kid": "test-key"},
        )

        claims = siwc_spike.validate_id_token_payload(
            token,
            jwks=jwks,
            client_id="oaiapp_test",
            nonce="nonce-123",
        )

        self.assertEqual(claims["sub"], "subject-123")
        with self.assertRaisesRegex(ValueError, "nonce mismatch"):
            siwc_spike.validate_id_token_payload(
                token,
                jwks=jwks,
                client_id="oaiapp_test",
                nonce="wrong",
            )

    def test_direct_plan_scope_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "chatgpt.tokens.use.direct"):
            siwc_spike.require_plan_scope("openid profile email")

        scopes = siwc_spike.require_plan_scope(
            "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
        )
        self.assertIn("chatgpt.tokens.use.direct", scopes)

    def test_stream_requires_response_completed(self) -> None:
        lines = [
            'data: {"type":"response.output_text.delta","delta":"Hello"}',
            'data: {"type":"response.output_text.delta","delta":", world"}',
            'data: {"type":"response.completed","response":{"usage":{"input_tokens":2,"output_tokens":3}}}',
        ]
        result = siwc_spike.parse_response_stream(lines)
        self.assertEqual(result.text, "Hello, world")
        self.assertTrue(result.completed)
        self.assertEqual(result.input_tokens, 2)
        self.assertEqual(result.output_tokens, 3)

    def test_tool_probe_payload_uses_supported_function_shape(self) -> None:
        payload = siwc_spike.build_tool_probe_payload("gpt-6.1-sol")

        self.assertFalse(payload["store"])
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["tool_choice"], "required")
        self.assertEqual(payload["tools"][0]["type"], "function")
        self.assertEqual(payload["tools"][0]["name"], "securo_spike_probe")
        self.assertNotIn("temperature", payload)
        self.assertNotIn("max_output_tokens", payload)

    def test_tool_probe_requires_exact_expected_call(self) -> None:
        valid = siwc_spike.StreamResult(
            text="",
            completed=True,
            tool_calls=(
                siwc_spike.FunctionCall(
                    call_id="call_123",
                    name="securo_spike_probe",
                    arguments={"proof": "live"},
                ),
            ),
        )
        siwc_spike.validate_tool_probe(valid)

        invalid_results = [
            siwc_spike.StreamResult(text="", completed=True),
            siwc_spike.StreamResult(
                text="",
                completed=True,
                tool_calls=(
                    siwc_spike.FunctionCall("", "securo_spike_probe", {"proof": "live"}),
                ),
            ),
            siwc_spike.StreamResult(
                text="",
                completed=True,
                tool_calls=(siwc_spike.FunctionCall("call_1", "other", {"proof": "live"}),),
            ),
            siwc_spike.StreamResult(
                text="",
                completed=True,
                tool_calls=(
                    siwc_spike.FunctionCall(
                        "call_1", "securo_spike_probe", {"proof": "wrong"}
                    ),
                ),
            ),
        ]
        for result in invalid_results:
            with self.subTest(result=result), self.assertRaisesRegex(RuntimeError, "exactly one"):
                siwc_spike.validate_tool_probe(result)

    def test_stream_collects_function_call_from_completed_output_item(self) -> None:
        result = siwc_spike.parse_response_stream(
            [
                'data: {"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_123","name":"securo_spike_probe","arguments":"{\\"proof\\":\\"ok\\"}"}}',
                'data: {"type":"response.completed","response":{"usage":{"input_tokens":9,"output_tokens":4}}}',
            ]
        )

        self.assertEqual(
            result.tool_calls,
            (
                siwc_spike.FunctionCall(
                    call_id="call_123",
                    name="securo_spike_probe",
                    arguments={"proof": "ok"},
                ),
            ),
        )

    def test_stream_ending_without_completed_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "without response.completed"):
            siwc_spike.parse_response_stream(
                ['data: {"type":"response.output_text.delta","delta":"partial"}']
            )

    def test_response_failed_surfaces_machine_code(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "subscription_sharing_usage_limit_exceeded"):
            siwc_spike.parse_response_stream(
                [
                    'data: {"type":"response.failed","response":{"error":{"code":"subscription_sharing_usage_limit_exceeded","message":"limit"}}}'
                ]
            )


class CredentialFileTests(unittest.TestCase):
    def test_credentials_are_written_atomically_with_owner_only_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credentials.json"
            siwc_spike.write_credentials(path, {"access_token": "secret", "scope": "openid"})

            self.assertEqual(json.loads(path.read_text()), {"access_token": "secret", "scope": "openid"})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


class RedirectSafetyTests(unittest.TestCase):
    def test_authenticated_request_does_not_follow_redirect(self) -> None:
        leaked_headers: list[str | None] = []

        class SinkHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                leaked_headers.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                return

        sink = ThreadingHTTPServer(("127.0.0.1", 0), SinkHandler)
        sink_thread = threading.Thread(target=sink.serve_forever, daemon=True)
        sink_thread.start()

        class RedirectHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(302)
                self.send_header(
                    "Location", f"http://127.0.0.1:{sink.server_address[1]}/capture"
                )
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                return

        redirect = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
        redirect_thread = threading.Thread(target=redirect.serve_forever, daemon=True)
        redirect_thread.start()
        try:
            request = Request(
                f"http://127.0.0.1:{redirect.server_address[1]}/start",
                headers={"Authorization": "Bearer must-not-leak"},
            )
            with self.assertRaisesRegex(RuntimeError, "HTTP 302"):
                siwc_spike.request_json(request)
            self.assertEqual(leaked_headers, [])
        finally:
            redirect.shutdown()
            sink.shutdown()
            redirect.server_close()
            sink.server_close()
            redirect_thread.join(timeout=2)
            sink_thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
