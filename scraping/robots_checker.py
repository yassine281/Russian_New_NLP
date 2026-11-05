"""Ethical web scraping compliance module for checking site robots.txt rules.

Theoretical Reference (Control Question 2):
Ethical web parsing requires respecting a website's `robots.txt` directives,
limiting request frequency to avoid server overload, identifying the client via
headers, and attributing data sources properly.
"""

from __future__ import annotations

import logging
from typing import Dict
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

from config import DEFAULT_HEADERS, REQUEST_TIMEOUT_SECONDS, USER_AGENT

logger = logging.getLogger(__name__)


class RobotsChecker:
    """Fetches, caches, and evaluates robots.txt permissions for target domains.

    Attributes:
        user_agent: User-Agent string evaluated against robots.txt rules.
        timeout: HTTP timeout in seconds when fetching robots.txt.
    """

    def __init__(
        self,
        user_agent: str = USER_AGENT,
        timeout: int = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize the RobotsChecker with an in-memory parser cache.

        Args:
            user_agent: User-Agent identifier string.
            timeout: Network timeout in seconds.
        """
        self.user_agent: str = user_agent
        self.timeout: int = timeout
        self._parsers: Dict[str, RobotFileParser] = {}
        self._unreachable_domains: set[str] = set()

    def can_fetch(self, url: str) -> bool:
        """Determine whether the given URL is permitted by the domain's robots.txt.

        Args:
            url: Full target URL to check.

        Returns:
            True if fetching is allowed (or if robots.txt is absent/permissive),
            False if explicitly disallowed by robots.txt.
        """
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            logger.warning("Invalid URL passed to RobotsChecker: %s", url)
            return False

        domain_root = f"{parsed.scheme}://{parsed.netloc}"
        if domain_root in self._unreachable_domains:
            return True

        parser = self._parsers.get(domain_root)
        if parser is None:
            parser = self._load_robots_parser(domain_root)
            if parser is None:
                self._unreachable_domains.add(domain_root)
                return True
            self._parsers[domain_root] = parser

        allowed_specific = parser.can_fetch(self.user_agent, url)
        allowed_wildcard = parser.can_fetch("*", url)
        is_allowed = allowed_specific and allowed_wildcard

        if not is_allowed:
            logger.info("Blocked by robots.txt policy: %s", url)
        return is_allowed

    def _load_robots_parser(self, domain_root: str) -> RobotFileParser | None:
        """Download and parse the robots.txt file for a domain root.

        Args:
            domain_root: Scheme and netloc (e.g., 'https://lenta.ru').

        Returns:
            Initialized RobotFileParser or None if robots.txt could not be fetched.
        """
        robots_url = f"{domain_root}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)

        try:
            response = requests.get(
                robots_url,
                headers=DEFAULT_HEADERS,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                lines = response.text.splitlines()
                parser.parse(lines)
                logger.debug("Loaded robots.txt from %s", robots_url)
                return parser
            if response.status_code in (401, 403):
                logger.warning(
                    "Access to %s returned HTTP %d; treating as restricted.",
                    robots_url,
                    response.status_code,
                )
                parser.disallow_all = True
                return parser
            logger.debug(
                "No restrictive robots.txt at %s (HTTP %d).",
                robots_url,
                response.status_code,
            )
            return None
        except requests.RequestException as exc:
            logger.warning("Could not fetch %s (%s).", robots_url, exc)
            return None