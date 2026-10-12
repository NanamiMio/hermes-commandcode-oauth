"""Credential-aware stdlib requests, including when Hermes is not installed."""
import urllib.parse
import urllib.request


def _origin(url):
    parsed = urllib.parse.urlsplit(url)
    scheme = parsed.scheme.lower()
    return scheme, (parsed.hostname or "").lower(), parsed.port or {"http": 80, "https": 443}.get(scheme)


class _CredentialRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, url):
        self.original_origin = _origin(url)

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and _origin(redirected.full_url) != self.original_origin:
            for name, _ in list(redirected.header_items()):
                if name.lower() not in ("accept", "user-agent"):
                    redirected.remove_header(name)
        return redirected


def open_credentialed(request, *, timeout):
    try:
        from hermes_cli.urllib_security import open_credentialed_url
    except ModuleNotFoundError as exc:
        if exc.name not in ("hermes_cli", "hermes_cli.urllib_security"):
            raise
        opener = urllib.request.build_opener(_CredentialRedirectHandler(request.full_url))
        return opener.open(request, timeout=timeout)
    return open_credentialed_url(request, timeout=timeout)
