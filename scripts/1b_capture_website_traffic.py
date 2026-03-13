"""Research-grade website traffic capture for fingerprinting experiments.

Captures network traffic from 100 websites across 10 categories using a
browser + tcpdump, producing CSV traces in the same format as the Alexa
voice command dataset: (index, time, size, direction).

Dataset specification:
    50-100 websites × 100 visits each (configurable via --sites flag)
    10 categories × 10 sites per category (100 total defined)
    Matches the scale of published WF papers:
        - Wang & Goldberg (USENIX Security 2014): 100 × 90
        - Panchenko et al. (NDSS 2016): 100 × 90
        - Sirinam et al. (CCS 2018): 95 × 1,000

Data quality measures (based on what the papers do):
    - DNS cache flush between visits (critical for non-Tor captures)
    - Quiet period before each capture (background traffic settles)
    - Fresh browser context per visit (no cookie/cache contamination)
    - Realistic browser profile (viewport, user-agent) to avoid bot detection
    - Post-load wait of 5 seconds (Wang et al. standard)
    - Screenshots for page-load verification
    - Automatic retry on failed captures (up to 3 attempts)
    - Resume capability (skips already-captured traces on restart)
    - Randomized visit order (prevents temporal artifacts)
    - JSON manifest with per-trace metadata
    - Minimum 50 packets per valid trace (literature threshold)

Requirements:
    - macOS with tcpdump (pre-installed)
    - sudo access (packet capture requires root)
    - Playwright + Chromium: uv run python -m playwright install chromium

Usage:
    cd traffic-fingerprinting

    # Dry run (2 sites × 2 visits) to verify everything works:
    sudo uv run python scripts/1b_capture_website_traffic.py --dry-run

    # Full capture (~22 hours, resume-safe):
    sudo uv run python scripts/1b_capture_website_traffic.py

    # Custom run (e.g., 10 sites × 20 visits):
    sudo uv run python scripts/1b_capture_website_traffic.py --sites 10 --visits 20

    # Resume after interruption — just re-run the same command:
    sudo uv run python scripts/1b_capture_website_traffic.py

Why sudo? tcpdump requires root to open the network interface in
promiscuous mode. This is standard for all packet capture research.

References:
    [1] Wang & Goldberg, "Effective Attacks and Provable Defenses for
        Website Fingerprinting," USENIX Security 2014.
    [2] Panchenko et al., "Website Fingerprinting at Internet Scale,"
        NDSS 2016.
    [3] Sirinam et al., "Deep Fingerprinting: Undermining Website
        Fingerprinting Defenses with Deep Learning," CCS 2018.
    [4] Kennedy et al., "I Know What You are Doing With Your Smart
        Speaker: Voice Command Fingerprinting," IEEE CNS 2019.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import random
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import dpkt
from playwright.sync_api import sync_playwright

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Page


# =============================================================================
#  Logging
# =============================================================================

LOG_FMT = "%(asctime)s [%(levelname)-5s] %(message)s"
LOG_DATE_FMT = "%H:%M:%S"

logger = logging.getLogger("wf_capture")


def setup_logging(log_file: Path, verbose: bool = False) -> None:
    """Configure logging to both console and file."""
    logger.setLevel(logging.DEBUG)

    # Console handler — INFO by default, DEBUG if verbose
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter(LOG_FMT, datefmt=LOG_DATE_FMT))
    logger.addHandler(console)

    # File handler — always DEBUG (full detail for post-hoc analysis)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)-5s] %(message)s")
    )
    logger.addHandler(file_handler)


# =============================================================================
#  Configuration
# =============================================================================

# Capture parameters — informed by published WF methodologies
VISITS_PER_SITE = 100
WAIT_AFTER_LOAD = 5        # Seconds post-load (Wang et al. use 5s) [1]
QUIET_PERIOD = 2           # Seconds before capture (let background traffic settle)
PAUSE_BETWEEN_VISITS = 3   # Seconds between visits (rate-limit protection)
NETWORK_INTERFACE = "en0"  # macOS WiFi interface
PAGE_TIMEOUT_MS = 30_000   # Playwright page load timeout
MIN_PACKETS = 50           # Minimum packets for a valid trace [2][3]
MAX_RETRIES = 3            # Retry attempts for failed captures
RANDOM_SEED = 42           # Reproducible visit order

# Realistic browser profile — avoids bot detection by major sites
BROWSER_VIEWPORT = {"width": 1440, "height": 900}
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

# Directory layout
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "collected" / "website_csv"
PCAP_DIR = BASE_DIR / "data" / "collected" / "pcap_raw"
SCREENSHOT_DIR = BASE_DIR / "data" / "collected" / "screenshots"
RESULTS_DIR = BASE_DIR / "results" / "website_dataset"
MANIFEST_PATH = RESULTS_DIR / "capture_manifest.json"
LOG_PATH = RESULTS_DIR / "capture.log"

# ---------------------------------------------------------------------------
#  100 Websites — 10 categories × 10 sites
#
#  Selection criteria:
#    - Publicly accessible without login (landing page loads content)
#    - HTTPS-only
#    - US-accessible, unlikely to hard-block headless browsers
#    - Within each category: mix of heavy pages (news, e-commerce) and
#      light pages (search, tools) for diverse traffic patterns
#    - Similar sites within categories to challenge the classifier
#      (e.g., CNN vs BBC vs Reuters in news)
# ---------------------------------------------------------------------------

WEBSITES: list[tuple[str, str, str]] = [
    # --- News (10) ---
    ("nytimes", "https://www.nytimes.com", "news"),
    ("cnn", "https://www.cnn.com", "news"),
    ("bbc", "https://www.bbc.com", "news"),
    ("reuters", "https://www.reuters.com", "news"),
    ("apnews", "https://apnews.com", "news"),
    ("foxnews", "https://www.foxnews.com", "news"),
    ("nbcnews", "https://www.nbcnews.com", "news"),
    ("usatoday", "https://www.usatoday.com", "news"),
    ("washingtonpost", "https://www.washingtonpost.com", "news"),
    ("npr", "https://www.npr.org", "news"),
    # --- Social Media (10) ---
    ("reddit", "https://www.reddit.com", "social"),
    ("twitter", "https://x.com", "social"),
    ("facebook", "https://www.facebook.com", "social"),
    ("linkedin", "https://www.linkedin.com", "social"),
    ("pinterest", "https://www.pinterest.com", "social"),
    ("tumblr", "https://www.tumblr.com", "social"),
    ("quora", "https://www.quora.com", "social"),
    ("mastodon", "https://mastodon.social", "social"),
    ("threads", "https://www.threads.net", "social"),
    ("discord", "https://discord.com", "social"),
    # --- E-commerce (10) ---
    ("amazon", "https://www.amazon.com", "ecommerce"),
    ("ebay", "https://www.ebay.com", "ecommerce"),
    ("walmart", "https://www.walmart.com", "ecommerce"),
    ("etsy", "https://www.etsy.com", "ecommerce"),
    ("target", "https://www.target.com", "ecommerce"),
    ("bestbuy", "https://www.bestbuy.com", "ecommerce"),
    ("homedepot", "https://www.homedepot.com", "ecommerce"),
    ("costco", "https://www.costco.com", "ecommerce"),
    ("wayfair", "https://www.wayfair.com", "ecommerce"),
    ("newegg", "https://www.newegg.com", "ecommerce"),
    # --- Tech / Developer (10) ---
    ("github", "https://github.com", "tech"),
    ("stackoverflow", "https://stackoverflow.com", "tech"),
    ("hackernews", "https://news.ycombinator.com", "tech"),
    ("gitlab", "https://gitlab.com", "tech"),
    ("devto", "https://dev.to", "tech"),
    ("medium", "https://medium.com", "tech"),
    ("techcrunch", "https://techcrunch.com", "tech"),
    ("arstechnica", "https://arstechnica.com", "tech"),
    ("theverge", "https://www.theverge.com", "tech"),
    ("wired", "https://www.wired.com", "tech"),
    # --- Entertainment (10) ---
    ("youtube", "https://www.youtube.com", "entertainment"),
    ("netflix", "https://www.netflix.com", "entertainment"),
    ("spotify", "https://www.spotify.com", "entertainment"),
    ("imdb", "https://www.imdb.com", "entertainment"),
    ("twitch", "https://www.twitch.tv", "entertainment"),
    ("hulu", "https://www.hulu.com", "entertainment"),
    ("rottentomatoes", "https://www.rottentomatoes.com", "entertainment"),
    ("soundcloud", "https://soundcloud.com", "entertainment"),
    ("disneyplus", "https://www.disneyplus.com", "entertainment"),
    ("crunchyroll", "https://www.crunchyroll.com", "entertainment"),
    # --- Education (10) ---
    ("wikipedia", "https://en.wikipedia.org", "education"),
    ("khanacademy", "https://www.khanacademy.org", "education"),
    ("coursera", "https://www.coursera.org", "education"),
    ("edx", "https://www.edx.org", "education"),
    ("mitocw", "https://ocw.mit.edu", "education"),
    ("britannica", "https://www.britannica.com", "education"),
    ("arxiv", "https://arxiv.org", "education"),
    ("wolframalpha", "https://www.wolframalpha.com", "education"),
    ("duolingo", "https://www.duolingo.com", "education"),
    ("udemy", "https://www.udemy.com", "education"),
    # --- Search / Tools (10) ---
    ("google", "https://www.google.com", "search"),
    ("bing", "https://www.bing.com", "search"),
    ("duckduckgo", "https://duckduckgo.com", "search"),
    ("yahoo", "https://www.yahoo.com", "search"),
    ("archive", "https://archive.org", "search"),
    ("googlescholar", "https://scholar.google.com", "search"),
    ("googlemaps", "https://maps.google.com", "search"),
    ("translate", "https://translate.google.com", "search"),
    ("speedtest", "https://www.speedtest.net", "search"),
    ("virustotal", "https://www.virustotal.com", "search"),
    # --- Sports (10) ---
    ("espn", "https://www.espn.com", "sports"),
    ("nba", "https://www.nba.com", "sports"),
    ("nfl", "https://www.nfl.com", "sports"),
    ("mlb", "https://www.mlb.com", "sports"),
    ("cbssports", "https://www.cbssports.com", "sports"),
    ("bleacherreport", "https://bleacherreport.com", "sports"),
    ("foxsports", "https://www.foxsports.com", "sports"),
    ("nhl", "https://www.nhl.com", "sports"),
    ("fifa", "https://www.fifa.com", "sports"),
    ("yahoosports", "https://sports.yahoo.com", "sports"),
    # --- Finance (10) ---
    ("yahoofinance", "https://finance.yahoo.com", "finance"),
    ("bloomberg", "https://www.bloomberg.com", "finance"),
    ("coinbase", "https://www.coinbase.com", "finance"),
    ("cnbc", "https://www.cnbc.com", "finance"),
    ("marketwatch", "https://www.marketwatch.com", "finance"),
    ("investopedia", "https://www.investopedia.com", "finance"),
    ("robinhood", "https://robinhood.com", "finance"),
    ("bankofamerica", "https://www.bankofamerica.com", "finance"),
    ("chase", "https://www.chase.com", "finance"),
    ("fidelity", "https://www.fidelity.com", "finance"),
    # --- Government / Reference (10) ---
    ("weathergov", "https://www.weather.gov", "government"),
    ("cdc", "https://www.cdc.gov", "government"),
    ("irs", "https://www.irs.gov", "government"),
    ("nih", "https://www.nih.gov", "government"),
    ("nasa", "https://www.nasa.gov", "government"),
    ("usps", "https://www.usps.com", "government"),
    ("whitehouse", "https://www.whitehouse.gov", "government"),
    ("census", "https://www.census.gov", "government"),
    ("fda", "https://www.fda.gov", "government"),
    ("loc", "https://www.loc.gov", "government"),
]


# =============================================================================
#  Data structures
# =============================================================================


@dataclass
class TraceMetadata:
    """Per-trace capture metadata for the manifest."""

    site_name: str
    url: str
    category: str
    visit_num: int
    n_packets: int
    n_outgoing: int
    n_incoming: int
    total_bytes: int
    duration_seconds: float
    capture_duration_seconds: float
    pcap_file: str
    csv_file: str
    screenshot_file: str
    sha256: str
    timestamp_utc: str
    attempt: int
    valid: bool
    failure_reason: str


@dataclass
class CaptureManifest:
    """Full manifest for the capture session — saved as JSON."""

    # Session info
    start_time_utc: str = ""
    end_time_utc: str = ""
    local_ip: str = ""
    interface: str = ""
    hostname: str = ""
    platform: str = ""
    python_version: str = ""

    # Configuration
    n_sites: int = 0
    visits_per_site: int = 0
    total_expected: int = 0
    wait_after_load_s: int = WAIT_AFTER_LOAD
    quiet_period_s: int = QUIET_PERIOD
    pause_between_visits_s: int = PAUSE_BETWEEN_VISITS
    min_packets: int = MIN_PACKETS
    max_retries: int = MAX_RETRIES
    random_seed: int = RANDOM_SEED

    # Counters
    total_captured: int = 0
    total_skipped: int = 0
    total_failed: int = 0
    total_retried: int = 0

    # Per-trace results
    traces: list[TraceMetadata] = field(default_factory=list)


# =============================================================================
#  Network utilities
# =============================================================================


def get_local_ip() -> str:
    """Detect this machine's local IP on the active network.

    Used to determine packet direction:
        source == local_ip  →  outgoing (+1)
        dest   == local_ip  →  incoming (-1)
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    finally:
        sock.close()


def flush_dns_cache() -> None:
    """Flush the macOS DNS cache between visits.

    Critical for non-Tor captures: stale DNS entries can route traffic
    through cached IPs, making different visits to the same site look
    artificially similar. Tor papers don't need this because the exit
    relay handles DNS resolution independently each time.
    """
    try:
        subprocess.run(
            ["dscacheutil", "-flushcache"],
            capture_output=True,
            timeout=5,
        )
        logger.debug("DNS cache flushed")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.debug("DNS cache flush skipped (not macOS or timeout)")


# =============================================================================
#  Packet capture (tcpdump)
# =============================================================================


def start_tcpdump(pcap_path: Path, interface: str) -> subprocess.Popen:
    """Start tcpdump capturing all TCP traffic on the interface.

    Captures all TCP rather than filtering by host IP because modern
    websites load resources from dozens of CDN domains with IPs that
    change between requests. Host-based BPF filters would miss the
    majority of page-load traffic.

    The -U flag ensures packet-buffered output so no data is lost
    if the process is terminated mid-capture.
    """
    cmd = [
        "tcpdump",
        "-i", interface,
        "-w", str(pcap_path),
        "-U",   # Packet-buffered (flush every packet, no data loss on kill)
        "-q",   # Quiet (less output to stderr)
        "tcp",  # TCP only — websites use TCP/TLS exclusively
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)  # tcpdump needs time to open the interface
    return proc


def stop_tcpdump(proc: subprocess.Popen) -> None:
    """Stop tcpdump gracefully, with forced kill as fallback."""
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        logger.warning("tcpdump did not stop gracefully, sending SIGKILL")
        proc.kill()
        proc.wait(timeout=3)


# =============================================================================
#  PCAP → CSV conversion
# =============================================================================


@dataclass
class ParseResult:
    """Result of parsing a PCAP file to CSV."""

    n_packets: int = 0
    n_outgoing: int = 0
    n_incoming: int = 0
    total_bytes: int = 0
    trace_duration: float = 0.0
    sha256: str = ""


def parse_pcap_to_csv(pcap_path: Path, csv_path: Path, local_ip: str) -> ParseResult:
    """Convert a PCAP file to the CSV trace format.

    Output CSV matches the Alexa dataset format exactly:
        ,time,size,direction
        0,0.000000,176.0,1.0
        1,0.000570,1500.0,-1.0
        ...

    Computes a SHA256 checksum of the output CSV for data integrity
    verification. Returns detailed packet statistics.
    """
    packets: list[tuple[float, int, int]] = []

    with open(pcap_path, "rb") as f:
        try:
            pcap_reader = dpkt.pcap.Reader(f)
        except ValueError:
            logger.debug("Empty or corrupt PCAP: %s", pcap_path.name)
            return ParseResult()

        for timestamp, buf in pcap_reader:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError):
                continue

            if not isinstance(eth.data, dpkt.ip.IP):
                continue

            ip_pkt = eth.data
            src_ip = socket.inet_ntoa(ip_pkt.src)
            direction = 1 if src_ip == local_ip else -1
            size = ip_pkt.len

            packets.append((timestamp, size, direction))

    if not packets:
        return ParseResult()

    # Normalize timestamps to start at 0
    t0 = packets[0][0]

    # Write CSV
    hasher = hashlib.sha256()
    with open(csv_path, "w", newline="") as f:
        header = ",time,size,direction\n"
        f.write(header)
        hasher.update(header.encode())

        for i, (ts, size, direction) in enumerate(packets):
            row = f"{i},{ts - t0:.6f},{size}.0,{direction}.0\n"
            f.write(row)
            hasher.update(row.encode())

    n_out = sum(1 for _, _, d in packets if d == 1)
    n_in = len(packets) - n_out
    total_bytes = sum(s for _, s, _ in packets)
    duration = packets[-1][0] - packets[0][0] if len(packets) > 1 else 0.0

    return ParseResult(
        n_packets=len(packets),
        n_outgoing=n_out,
        n_incoming=n_in,
        total_bytes=total_bytes,
        trace_duration=round(duration, 4),
        sha256=hasher.hexdigest(),
    )


# =============================================================================
#  Browser automation
# =============================================================================


def create_browser_context(browser: Browser):
    """Create a fresh browser context with a realistic profile.

    Each visit uses a new context (isolated cookies, cache, storage)
    matching the methodology of all published WF papers [1][2][3].

    The viewport and user-agent are set to realistic values to avoid
    bot-detection systems on major websites. Pure headless with default
    Playwright headers triggers CAPTCHAs on Amazon, Cloudflare, etc.
    """
    return browser.new_context(
        viewport=BROWSER_VIEWPORT,
        user_agent=BROWSER_USER_AGENT,
        locale="en-US",
        timezone_id="America/Chicago",
        java_script_enabled=True,
    )


def visit_and_capture_screenshot(
    url: str,
    page: Page,
    screenshot_path: Path,
) -> bool:
    """Visit a URL, wait for full load, and take a verification screenshot.

    The screenshot serves two purposes:
    1. Post-hoc verification that the page actually loaded (not a
       CAPTCHA, error page, or blank screen)
    2. Visual documentation of what content was visible at capture time

    Returns True if the page loaded (even partially), False on total failure.
    """
    page_loaded = False
    try:
        page.goto(url, wait_until="networkidle", timeout=PAGE_TIMEOUT_MS)
        page_loaded = True
    except Exception as exc:
        # Timeout or navigation error — still capture whatever loaded
        logger.debug("Page load issue for %s: %s", url, type(exc).__name__)
        page_loaded = True  # Partial load is still usable data

    # Post-load wait — Wang et al. use 5 seconds to capture trailing
    # packets (analytics pings, lazy-loaded images, WebSocket heartbeats)
    time.sleep(WAIT_AFTER_LOAD)

    # Take verification screenshot
    try:
        page.screenshot(path=str(screenshot_path), full_page=False)
    except Exception:
        logger.debug("Screenshot failed for %s", url)

    return page_loaded


# =============================================================================
#  Visit scheduling
# =============================================================================


def build_visit_schedule(
    websites: list[tuple[str, str, str]],
    visits_per_site: int,
    seed: int,
) -> list[tuple[str, str, str, int]]:
    """Build a randomized schedule of all (site, visit_num) pairs.

    Randomization prevents temporal artifacts: without it, all 100
    visits to google.com happen consecutively (e.g., 2:00 AM–2:15 AM),
    and network conditions during that window become a confound.
    Shuffling distributes each site's visits across the full capture
    period, so the classifier must learn traffic shape rather than
    time-of-day patterns.
    """
    schedule: list[tuple[str, str, str, int]] = []
    for site_name, url, category in websites:
        for visit_num in range(1, visits_per_site + 1):
            schedule.append((site_name, url, category, visit_num))

    rng = random.Random(seed)
    rng.shuffle(schedule)
    return schedule


# =============================================================================
#  Formatting helpers
# =============================================================================


def format_eta(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 0:
        return "0s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def system_info() -> dict[str, str]:
    """Collect system metadata for the manifest."""
    return {
        "hostname": platform.node(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": platform.python_version(),
    }


# =============================================================================
#  Single capture with retry
# =============================================================================


def capture_single_trace(
    site_name: str,
    url: str,
    category: str,
    visit_num: int,
    browser: Browser,
    local_ip: str,
    interface: str,
) -> TraceMetadata:
    """Capture a single website trace with automatic retry.

    On failure (< MIN_PACKETS packets), retries up to MAX_RETRIES times
    with exponential backoff. Each retry uses a fresh browser context
    and DNS cache flush.

    Returns a TraceMetadata with all capture details.
    """
    pcap_file = PCAP_DIR / f"{site_name}_{visit_num}.pcap"
    csv_file = OUTPUT_DIR / f"{site_name}_{visit_num}.csv"
    screenshot_file = SCREENSHOT_DIR / f"{site_name}_{visit_num}.png"

    best_result: ParseResult | None = None
    best_attempt = 0
    failure_reason = ""

    for attempt in range(1, MAX_RETRIES + 1):
        capture_start = time.time()

        # Flush DNS cache to avoid stale routing
        flush_dns_cache()

        # Quiet period — let background traffic settle
        time.sleep(QUIET_PERIOD)

        # Fresh browser context
        context = create_browser_context(browser)
        page = context.new_page()

        try:
            # Start packet capture
            tcpdump_proc = start_tcpdump(pcap_file, interface)

            # Visit the website and take screenshot
            visit_and_capture_screenshot(url, page, screenshot_file)

            # Stop capture
            stop_tcpdump(tcpdump_proc)

        except Exception as exc:
            failure_reason = f"Capture error: {type(exc).__name__}: {exc}"
            logger.warning(
                "Capture error for %s v%d attempt %d: %s",
                site_name, visit_num, attempt, failure_reason,
            )
        finally:
            # Always clean up the browser context
            try:
                context.close()
            except Exception:
                pass

        # Parse PCAP → CSV
        result = parse_pcap_to_csv(pcap_file, csv_file, local_ip)
        capture_duration = time.time() - capture_start

        if result.n_packets >= MIN_PACKETS:
            # Success
            return TraceMetadata(
                site_name=site_name,
                url=url,
                category=category,
                visit_num=visit_num,
                n_packets=result.n_packets,
                n_outgoing=result.n_outgoing,
                n_incoming=result.n_incoming,
                total_bytes=result.total_bytes,
                duration_seconds=result.trace_duration,
                capture_duration_seconds=round(capture_duration, 2),
                pcap_file=pcap_file.name,
                csv_file=csv_file.name,
                screenshot_file=screenshot_file.name,
                sha256=result.sha256,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                attempt=attempt,
                valid=True,
                failure_reason="",
            )

        # Not enough packets — record best attempt and retry
        failure_reason = f"Only {result.n_packets} packets (min: {MIN_PACKETS})"
        if best_result is None or result.n_packets > best_result.n_packets:
            best_result = result
            best_attempt = attempt

        if attempt < MAX_RETRIES:
            backoff = 2 ** attempt
            logger.debug(
                "Retry %d/%d for %s v%d (%d packets < %d min), "
                "waiting %ds",
                attempt, MAX_RETRIES, site_name, visit_num,
                result.n_packets, MIN_PACKETS, backoff,
            )
            time.sleep(backoff)

    # All retries exhausted — return best attempt
    if best_result is None:
        best_result = ParseResult()

    return TraceMetadata(
        site_name=site_name,
        url=url,
        category=category,
        visit_num=visit_num,
        n_packets=best_result.n_packets,
        n_outgoing=best_result.n_outgoing,
        n_incoming=best_result.n_incoming,
        total_bytes=best_result.total_bytes,
        duration_seconds=best_result.trace_duration,
        capture_duration_seconds=0.0,
        pcap_file=pcap_file.name,
        csv_file=csv_file.name,
        screenshot_file=screenshot_file.name if screenshot_file.exists() else "",
        sha256=best_result.sha256,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        attempt=best_attempt,
        valid=False,
        failure_reason=failure_reason,
    )


# =============================================================================
#  Manifest I/O
# =============================================================================


def save_manifest(manifest: CaptureManifest) -> None:
    """Save the capture manifest to JSON."""
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2, ensure_ascii=False)


def load_manifest() -> CaptureManifest | None:
    """Load an existing manifest for resume, if available."""
    if not MANIFEST_PATH.exists():
        return None
    try:
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            data = json.load(f)
        # Reconstruct dataclass from dict
        traces = [TraceMetadata(**t) for t in data.pop("traces", [])]
        manifest = CaptureManifest(**data)
        manifest.traces = traces
        return manifest
    except (json.JSONDecodeError, TypeError, KeyError):
        logger.warning("Could not load existing manifest, starting fresh")
        return None


# =============================================================================
#  Main
# =============================================================================


def main() -> None:
    """Entry point: parse args, set up, and run the capture loop."""
    parser = argparse.ArgumentParser(
        description="Research-grade website traffic capture for fingerprinting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Quick test: 2 sites × 2 visits (verify setup)",
    )
    parser.add_argument(
        "--sites", type=int, default=None,
        help="Number of sites to capture (default: all 100)",
    )
    parser.add_argument(
        "--visits", type=int, default=None,
        help="Visits per site (default: 100)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show DEBUG-level log messages on console",
    )
    parser.add_argument(
        "--interface", type=str, default=NETWORK_INTERFACE,
        help=f"Network interface (default: {NETWORK_INTERFACE})",
    )
    args = parser.parse_args()

    # Create all output directories
    for d in [OUTPUT_DIR, PCAP_DIR, SCREENSHOT_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Set up logging
    setup_logging(LOG_PATH, verbose=args.verbose)

    # Determine capture parameters
    if args.dry_run:
        websites = WEBSITES[:2]
        visits_per_site = 2
        logger.info("DRY RUN MODE — 2 sites × 2 visits")
    else:
        websites = WEBSITES[: args.sites] if args.sites else WEBSITES
        visits_per_site = args.visits if args.visits else VISITS_PER_SITE

    interface = args.interface
    local_ip = get_local_ip()
    total_expected = len(websites) * visits_per_site
    sysinfo = system_info()

    logger.info("=" * 60)
    logger.info("  Website Traffic Capture — Fingerprinting Dataset")
    logger.info("=" * 60)
    logger.info("  Host:          %s", sysinfo["hostname"])
    logger.info("  Platform:      %s", sysinfo["platform"])
    logger.info("  Local IP:      %s", local_ip)
    logger.info("  Interface:     %s", interface)
    logger.info("  Sites:         %d", len(websites))
    logger.info("  Visits/site:   %d", visits_per_site)
    logger.info("  Total traces:  %d", total_expected)
    logger.info("  Wait/load:     %ds", WAIT_AFTER_LOAD)
    logger.info("  Quiet period:  %ds", QUIET_PERIOD)
    logger.info("  Min packets:   %d", MIN_PACKETS)
    logger.info("  Max retries:   %d", MAX_RETRIES)
    logger.info("  Output:        %s", OUTPUT_DIR)
    logger.info("  Log:           %s", LOG_PATH)
    logger.info("=" * 60)

    # Build randomized schedule
    schedule = build_visit_schedule(websites, visits_per_site, RANDOM_SEED)

    # Resume support — skip already-captured traces
    existing_csvs = {f.name for f in OUTPUT_DIR.glob("*.csv")}
    remaining: list[tuple[str, str, str, int]] = []
    skipped = 0
    for site_name, url, category, visit_num in schedule:
        csv_name = f"{site_name}_{visit_num}.csv"
        if csv_name in existing_csvs:
            skipped += 1
        else:
            remaining.append((site_name, url, category, visit_num))

    if skipped > 0:
        logger.info(
            "RESUMING: %d already captured, %d remaining",
            skipped, len(remaining),
        )
    if not remaining:
        logger.info("All %d traces already captured. Nothing to do.", total_expected)
        return

    est_seconds = len(remaining) * (WAIT_AFTER_LOAD + QUIET_PERIOD + PAUSE_BETWEEN_VISITS + 4)
    logger.info("Estimated time: %s", format_eta(est_seconds))
    logger.info("")

    # Initialize manifest
    manifest = CaptureManifest(
        start_time_utc=datetime.now(timezone.utc).isoformat(),
        local_ip=local_ip,
        interface=interface,
        hostname=sysinfo["hostname"],
        platform=sysinfo["platform"],
        python_version=sysinfo["python_version"],
        n_sites=len(websites),
        visits_per_site=visits_per_site,
        total_expected=total_expected,
        total_skipped=skipped,
    )

    # Graceful shutdown on Ctrl+C
    shutdown_requested = False

    def signal_handler(signum, frame):  # noqa: ARG001
        nonlocal shutdown_requested
        if shutdown_requested:
            logger.warning("Force quit — exiting immediately")
            sys.exit(1)
        shutdown_requested = True
        logger.info("")
        logger.info("Shutdown requested (Ctrl+C). Finishing current capture...")
        logger.info("Press Ctrl+C again to force quit.")

    signal.signal(signal.SIGINT, signal_handler)

    # Main capture loop
    captured = 0
    failed = 0
    retried = 0
    loop_start = time.time()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)

        for i, (site_name, url, category, visit_num) in enumerate(remaining):
            if shutdown_requested:
                logger.info("Stopping after %d captures (shutdown requested)", captured)
                break

            # Capture with retry
            trace = capture_single_trace(
                site_name, url, category, visit_num,
                browser, local_ip, interface,
            )
            manifest.traces.append(trace)

            if trace.valid:
                captured += 1
                if trace.attempt > 1:
                    retried += 1
            else:
                failed += 1

            # Progress
            done = i + 1
            total = len(remaining)
            elapsed = time.time() - loop_start
            avg = elapsed / done
            eta = avg * (total - done)
            pct = done / total * 100

            status = "OK" if trace.valid else "FAIL"
            retry_tag = f" (attempt {trace.attempt})" if trace.attempt > 1 else ""
            logger.info(
                "[%5d/%d] %5.1f%%  %-20s v%-3d  %5d pkts  [%s]%s  ETA %s",
                done, total, pct,
                site_name, visit_num,
                trace.n_packets, status, retry_tag,
                format_eta(eta),
            )

            # Save manifest every 25 captures (crash recovery)
            if done % 25 == 0:
                manifest.total_captured = captured + skipped
                manifest.total_failed = failed
                manifest.total_retried = retried
                save_manifest(manifest)
                logger.debug("Manifest checkpoint saved (%d traces)", done)

            # Pause between visits
            time.sleep(PAUSE_BETWEEN_VISITS)

        browser.close()

    # Final manifest
    manifest.end_time_utc = datetime.now(timezone.utc).isoformat()
    manifest.total_captured = captured + skipped
    manifest.total_failed = failed
    manifest.total_retried = retried
    save_manifest(manifest)

    # Summary
    total_elapsed = time.time() - loop_start
    csv_count = len(list(OUTPUT_DIR.glob("*.csv")))

    logger.info("")
    logger.info("=" * 60)
    logger.info("  CAPTURE COMPLETE")
    logger.info("=" * 60)
    logger.info("  Total time:     %s", format_eta(total_elapsed))
    logger.info("  CSV files:      %d / %d expected", csv_count, total_expected)
    logger.info("  New captures:   %d", captured)
    logger.info("  Skipped:        %d (resumed)", skipped)
    logger.info("  Failed:         %d (< %d packets)", failed, MIN_PACKETS)
    logger.info("  Retried:        %d (succeeded on retry)", retried)
    logger.info("  Manifest:       %s", MANIFEST_PATH)
    logger.info("  Log:            %s", LOG_PATH)
    logger.info("=" * 60)

    if failed > 0:
        logger.warning("")
        logger.warning(
            "%d traces below %d-packet threshold:", failed, MIN_PACKETS,
        )
        failed_traces = [t for t in manifest.traces if not t.valid]
        for t in failed_traces[:15]:
            logger.warning(
                "  %-25s v%-3d  %4d pkts  %s",
                t.site_name, t.visit_num, t.n_packets, t.failure_reason,
            )
        if len(failed_traces) > 15:
            logger.warning("  ... and %d more (see manifest)", len(failed_traces) - 15)


if __name__ == "__main__":
    main()
