from urllib.parse import urlsplit


def validate_notifier_url(url: str, *, allow_insecure_http: bool = False) -> str:
    """Validate a configured notification endpoint without exposing its path."""
    if not isinstance(url, str) or not url:
        raise ValueError("notifier URL must be a non-empty string")
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in url):
        raise ValueError("notifier URL must not contain whitespace or control characters")
    try:
        parsed = urlsplit(url)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("notifier URL is malformed") from exc
    allowed = {"https"}
    if allow_insecure_http:
        allowed.add("http")
    if parsed.scheme.casefold() not in allowed:
        raise ValueError("notifier URL must use HTTPS")
    if parsed.hostname is None:
        raise ValueError("notifier URL must include a host")
    try:
        parsed.hostname.encode("idna")
    except UnicodeError as exc:
        raise ValueError("notifier URL host is invalid") from exc
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("notifier URL must not contain user credentials")
    if parsed.fragment:
        raise ValueError("notifier URL must not contain a fragment")
    return url
