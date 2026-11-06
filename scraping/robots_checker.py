"""Модуль проверки файла robots.txt для этичного парсинга."""

from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser


class RobotsChecker:
    """Проверяет разрешение на парсинг страниц в соответствии с robots.txt."""

    def __init__(self) -> None:
        self._parsers = {}

    def can_fetch(self, url: str) -> bool:
        """Возвращает True, если парсинг URL разрешен правилами сайта."""
        parsed = urlparse(url)
        if not parsed.netloc:
            return True
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        if base_url not in self._parsers:
            parser = RobotFileParser()
            parser.set_url(f"{base_url}/robots.txt")
            try:
                parser.read()
            except Exception:
                pass
            self._parsers[base_url] = parser
        try:
            return bool(self._parsers[base_url].can_fetch("*", url))
        except Exception:
            return True