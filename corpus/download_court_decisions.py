"""Downloads Hungarian court decisions for the large-scale RAG evaluation corpus.

Source: eakta.birosag.hu's "Bírósági Határozatok Gyűjteménye" (anonymized court
decisions). There is no public API documentation for this site -- the search
endpoint below was reverse-engineered from the page's embedded JavaScript
(the "ENCO.Grid" component the site's search form is built on) and verified
against live responses.

Run it with (from the repo root):
    uv run python corpus/download_court_decisions.py [options]

Example: real estate law, "polgári" college, both judgment types, 2010-2024:
    uv run python corpus/download_court_decisions.py \\
        --kollegium polgári \\
        --decision-types Ítélet Végzés \\
        --keywords ingatlan tulajdonjog ingatlan-nyilvántartás adásvétel bérlet társasház \\
        --year-from 2010 --year-to 2024 \\
        --max-documents 10000

Output (inside --out-dir, default corpus/):
    raw/<court>__<case_number>.{rtf,docx}   -- one file per decision, native format
                                                (--format pdf downloads .pdf instead;
                                                the native endpoint returns RTF or DOCX
                                                depending on the source court/era, so
                                                both extensions can appear side by side)
    meta.csv                                -- case_number, court, year, decision_type,
                                                source_url, file_name, status

Known data-quality limitation: the search endpoint only exposes the decision's
YEAR (HatarozatEve), not its exact date -- the "year" column in meta.csv is a
year, not a full date. The exact date only appears inside the document text
itself, if it's needed later it has to be extracted from there.

The site's "HatarozatFajta" (decision type) filter accepts exactly one value
per search -- there is no way to ask for "Ítélet OR Végzés" in a single
request. This script runs one search per (decision_type, keyword) combination
and deduplicates results across all of them by (court, case_number).
"""

import argparse
import csv
import logging
import random
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import requests

# Ensure project root is on sys.path so `import retry` resolves when this
# script is run directly (`uv run python corpus/download_court_decisions.py`),
# not just when imported as a package.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from retry_policy import TransientAPIError, retry_on_transient_error

logger = logging.getLogger(__name__)

BASE_URL = "https://eakta.birosag.hu"
SEARCH_PAGE_URL = f"{BASE_URL}/anonimizalt-hatarozatok"
SEARCH_URL = f"{BASE_URL}/AnonimizaltHatarozat/Search?Area="
PDF_DOWNLOAD_URL = f"{BASE_URL}/anonimizalt-hatarozat-pdf/"
RTF_DOWNLOAD_URL = f"{BASE_URL}/AnonimizaltHatarozat/DownloadHatarozatMobile"

# The site's own accepted values for these two select fields (Hungarian, taken
# directly from the search form's <option> elements -- not translatable).
DEFAULT_DECISION_TYPES = ["Ítélet", "Végzés"]
DEFAULT_KEYWORDS = [
    "ingatlan",
    "tulajdonjog",
    "ingatlan-nyilvántartás",
    "adásvétel",
    "bérlet",
    "társasház",
]
# The "how does a keyword match" mode (see the KeresoSzoOperatorDropdown
# options in the site's own HTML): "Osszes" = all words of the keyword phrase
# must occur, the closest match to a plain keyword search.
KEYWORD_MATCH_MODE = "Osszes"

USER_AGENT = (
    "docs-agent-corpus-collector/1.0 (research corpus collection; contact via repo)"
)

# The "native format" download endpoint returns whichever format the source
# document was originally filed in -- RTF or DOCX, depending on the court/era
# -- not necessarily RTF. The real extension has to be read back from the
# response rather than assumed from --format.
NATIVE_FORMAT_EXTENSIONS = ["rtf", "docx"]
CONTENT_TYPE_EXTENSIONS = {
    "application/rtf": "rtf",
    "text/rtf": "rtf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "doc",
    "application/pdf": "pdf",
}

# Safety valve: stop paginating a single (decision_type, keyword) search after
# this many result pages, in case the server ever reports a result count that
# doesn't match how many pages it's actually willing to return.
MAX_PAGES_PER_SEARCH = 2000


@dataclass
class DownloadConfig:
    """Resolved CLI configuration for one corpus-download run."""

    kollegium: str
    decision_types: list[str]
    keywords: list[str]
    year_from: int
    year_to: int
    max_documents: int
    page_size: int
    delay_min: float
    delay_max: float
    max_retries: int
    out_dir: Path
    file_format: str  # "pdf" or "rtf"

    @property
    def raw_dir(self) -> Path:
        return self.out_dir / "raw"

    @property
    def meta_csv_path(self) -> Path:
        return self.out_dir / "meta.csv"


@dataclass
class DecisionRecord:
    """One search-result row, enough to build both a download URL and a meta.csv row."""

    case_number: str  # "Azonosito", e.g. "Pfv.20721/2013/25"
    court: str  # "MeghozoBirosag", e.g. "Kúria"
    year: int  # "HatarozatEve" -- see the module docstring's date caveat
    decision_type: str  # which --decision-types search this came from
    index_id: str  # "IndexId", required by the download URL


def create_session() -> requests.Session:
    """Creates a requests.Session carrying the session cookie the search endpoint requires.

    Returns:
        A requests.Session that has already visited the search page once, so
        it holds a valid session cookie for subsequent search/download calls.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    response = session.get(SEARCH_PAGE_URL, timeout=30)
    response.raise_for_status()
    return session


def sleep_politely(config: DownloadConfig) -> None:
    """Sleeps a random duration within the configured rate-limit window."""
    time.sleep(random.uniform(config.delay_min, config.delay_max))


def request_with_retries(
    session: requests.Session, method: str, url: str, config: DownloadConfig, **kwargs
) -> requests.Response | None:
    """Issues an HTTP request, retrying with backoff on network errors, 429, or 5xx.

    Uses :func:`retry.retry_on_transient_error` for the retry/backoff
    mechanics (shared with the embedding drivers in
    ``drivers/embedding.py``), applied inline since ``config.max_retries``
    is only known at call time, not decoration time.

    Args:
        method: HTTP method ("GET" or "POST").
        url: Full request URL.
        kwargs: Passed through to requests.Session.request (e.g. data, headers).

    Returns:
        The successful Response, or None if every retry was exhausted.
    """

    @retry_on_transient_error(max_attempts=config.max_retries)
    def _do_request() -> requests.Response:
        try:
            response = session.request(method, url, timeout=60, **kwargs)
        except requests.RequestException as exc:
            raise TransientAPIError(str(exc)) from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise TransientAPIError(f"{url} returned status {response.status_code}")

        return response

    try:
        return _do_request()
    except TransientAPIError:
        logger.error("Giving up on %s after %d attempts", url, config.max_retries)
        return None


def search_page(
    session: requests.Session,
    config: DownloadConfig,
    decision_type: str,
    keyword: str,
    start_index: int,
) -> dict | None:
    """Runs one page of the search and returns the parsed JSON response.

    Returns:
        The response payload (with "List"/"Count"/"Success" keys), or None if
        the request failed or the server reported Success=false.
    """
    data = {
        "Kollegium": config.kollegium,
        "HatarozatFajta": decision_type,
        "MeghozatalIdejeTol": str(config.year_from),
        "MeghozatalIdejeIg": str(config.year_to),
        "ResultSortExpression": "HatarozatEveCsokkeno",
        "ResultCount": str(config.page_size),
        "ResultStartIndex": str(start_index),
        "KeresoSzavak[]": keyword,
        "KeresoSzoOperatorok[]": KEYWORD_MATCH_MODE,
    }
    headers = {
        "X-Requested-With": "XMLHttpRequest",
        "Referer": SEARCH_PAGE_URL,
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    }
    response = request_with_retries(
        session, "POST", SEARCH_URL, config, data=data, headers=headers
    )
    if response is None:
        return None

    try:
        payload = response.json()
    except ValueError:
        logger.error(
            "Non-JSON response from search endpoint (status %d)", response.status_code
        )
        return None

    if not payload.get("Success"):
        logger.error("Search request rejected by server: %s", payload.get("Message"))
        return None

    return payload


def iter_search_results(
    session: requests.Session, config: DownloadConfig, decision_type: str, keyword: str
):
    """Yields a DecisionRecord for every result of one (decision_type, keyword) search, paginating as needed."""
    start_index = 0
    for _ in range(MAX_PAGES_PER_SEARCH):
        payload = search_page(session, config, decision_type, keyword, start_index)
        if payload is None:
            return

        items = payload.get("List") or []
        if not items:
            return

        for item in items:
            yield DecisionRecord(
                case_number=item["Azonosito"],
                court=item["MeghozoBirosag"],
                year=item["HatarozatEve"],
                decision_type=decision_type,
                index_id=item["IndexId"],
            )

        start_index += config.page_size
        sleep_politely(config)

    logger.warning(
        "Hit MAX_PAGES_PER_SEARCH=%d for decision_type=%s keyword=%r -- stopping this search early",
        MAX_PAGES_PER_SEARCH,
        decision_type,
        keyword,
    )


def slugify(value: str) -> str:
    """Converts a Hungarian court name or case number into a filesystem-safe ASCII slug."""
    normalized = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    )
    return re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_")


def download_url_for(config: DownloadConfig, record: DecisionRecord) -> str:
    """Builds the direct download/view URL for one decision, in the configured file format."""
    if config.file_format == "pdf":
        return f"{PDF_DOWNLOAD_URL}?birosagName={record.court}&ugyszam={record.case_number}&azonosito={record.index_id}"
    return (
        f"{RTF_DOWNLOAD_URL}?Area=&birosagName={record.court}"
        f"&ugyszam={record.case_number}&azonosito={record.index_id}"
    )


def _extension_from_response(response: requests.Response, fallback: str) -> str:
    """Determines the real file extension from the response's Content-Type/Content-Disposition."""
    content_type = (
        response.headers.get("Content-Type", "").split(";")[0].strip().lower()
    )
    if content_type in CONTENT_TYPE_EXTENSIONS:
        return CONTENT_TYPE_EXTENSIONS[content_type]

    content_disposition = response.headers.get("Content-Disposition", "")
    match = re.search(r'filename=\s*"?([^";]+)"?', content_disposition)
    if match:
        suffix = Path(match.group(1)).suffix.lstrip(".").lower()
        if suffix:
            return suffix

    return fallback


def download_decision(
    session: requests.Session, config: DownloadConfig, record: DecisionRecord
) -> tuple[Path, str]:
    """Downloads one decision's document to raw_dir, skipping it if already present on disk.

    For --format rtf, the actual bytes may turn out to be RTF or DOCX (see
    NATIVE_FORMAT_EXTENSIONS) -- both candidate filenames are checked before
    downloading, since the real extension isn't known until the response
    arrives.

    Returns:
        A (file_path, status) tuple, status is one of
        "already_downloaded" / "downloaded" / "error".
    """
    base_name = f"{slugify(record.court)}__{slugify(record.case_number)}"
    candidate_extensions = (
        ["pdf"] if config.file_format == "pdf" else NATIVE_FORMAT_EXTENSIONS
    )

    for extension in candidate_extensions:
        existing_path = config.raw_dir / f"{base_name}.{extension}"
        if existing_path.exists():
            return existing_path, "already_downloaded"

    url = download_url_for(config, record)
    response = request_with_retries(
        session, "GET", url, config, headers={"Referer": SEARCH_PAGE_URL}
    )
    fallback_path = config.raw_dir / f"{base_name}.{candidate_extensions[0]}"
    if response is None or response.status_code != 200:
        return fallback_path, "error"

    extension = _extension_from_response(response, fallback=candidate_extensions[0])
    file_path = config.raw_dir / f"{base_name}.{extension}"
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and rename atomically, so a crash/interrupt mid-write
    # (power loss, disk full, kill -9) can never leave a truncated file behind
    # that a later resume would mistake for a complete, already-downloaded one.
    temp_path = file_path.with_suffix(file_path.suffix + ".part")
    temp_path.write_bytes(response.content)
    temp_path.replace(file_path)
    return file_path, "downloaded"


SUCCESSFUL_STATUSES = {"downloaded", "already_downloaded"}


def load_already_processed(meta_csv_path: Path) -> set[tuple[str, str]]:
    """Reads an existing meta.csv, if any, to determine what a re-run can skip.

    meta.csv is append-only, so the same (court, case_number) can appear more
    than once (e.g. an "error" row from an interrupted run, followed later by
    a successful "downloaded" row after a resume). Only the LAST recorded
    status per key decides whether it's treated as done -- a key whose latest
    status is "error" is deliberately left out, so the next run retries it
    instead of skipping it forever.
    """
    if not meta_csv_path.exists():
        return set()

    latest_status: dict[tuple[str, str], str] = {}
    with meta_csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["court"], row["case_number"])
            latest_status[key] = row["status"]

    return {
        key for key, status in latest_status.items() if status in SUCCESSFUL_STATUSES
    }


def run(config: DownloadConfig) -> None:
    """Runs every configured (decision_type, keyword) search, downloading and recording each unique decision."""
    config.out_dir.mkdir(parents=True, exist_ok=True)
    config.raw_dir.mkdir(parents=True, exist_ok=True)

    already_processed = load_already_processed(config.meta_csv_path)
    write_header = not config.meta_csv_path.exists()
    session = create_session()

    seen_this_run: set[tuple[str, str]] = set()
    downloaded_count = 0

    with config.meta_csv_path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(
                [
                    "case_number",
                    "court",
                    "year",
                    "decision_type",
                    "source_url",
                    "file_name",
                    "status",
                ]
            )

        for decision_type in config.decision_types:
            for keyword in config.keywords:
                logger.info(
                    "Searching decision_type=%s keyword=%r", decision_type, keyword
                )

                for record in iter_search_results(
                    session, config, decision_type, keyword
                ):
                    key = (record.court, record.case_number)
                    if key in already_processed or key in seen_this_run:
                        continue
                    seen_this_run.add(key)

                    file_path, status = download_decision(session, config, record)
                    writer.writerow(
                        [
                            record.case_number,
                            record.court,
                            record.year,
                            record.decision_type,
                            download_url_for(config, record),
                            file_path.name,
                            status,
                        ]
                    )
                    csv_file.flush()

                    if status == "downloaded":
                        downloaded_count += 1
                        logger.info(
                            "[%d/%d] downloaded %s (%s, %d)",
                            downloaded_count,
                            config.max_documents,
                            record.case_number,
                            record.court,
                            record.year,
                        )
                    elif status == "error":
                        logger.warning(
                            "Failed to download %s (%s)",
                            record.case_number,
                            record.court,
                        )

                    sleep_politely(config)

                    if downloaded_count >= config.max_documents:
                        logger.info(
                            "Reached --max-documents=%d, stopping.",
                            config.max_documents,
                        )
                        return

    logger.info("Done. %d new documents downloaded.", downloaded_count)


def parse_args() -> DownloadConfig:
    """Parses CLI arguments into a DownloadConfig."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--kollegium", default="polgári", help="Kollegium filter value, e.g. 'polgári'."
    )
    parser.add_argument(
        "--decision-types",
        nargs="+",
        default=DEFAULT_DECISION_TYPES,
        help="HatarozatFajta values to search one at a time (the site allows only one per search).",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=DEFAULT_KEYWORDS,
        help="Keywords searched one at a time; results are merged and deduplicated across all of them.",
    )
    parser.add_argument("--year-from", type=int, default=2010)
    parser.add_argument("--year-to", type=int, default=2024)
    parser.add_argument(
        "--max-documents",
        type=int,
        default=10000,
        help="Stop after downloading this many new documents.",
    )
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument(
        "--delay-min", type=float, default=0.5, help="Minimum seconds between requests."
    )
    parser.add_argument(
        "--delay-max", type=float, default=1.0, help="Maximum seconds between requests."
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory to write raw/ and meta.csv into (default: this script's own directory).",
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "rtf"],
        default="rtf",
        dest="file_format",
        help="Document format to download. RTF's text extracts more cleanly than PDF's "
        "coordinate-based layout and has no extractor yet on the ingestion side (default: rtf).",
    )
    args = parser.parse_args()

    return DownloadConfig(
        kollegium=args.kollegium,
        decision_types=args.decision_types,
        keywords=args.keywords,
        year_from=args.year_from,
        year_to=args.year_to,
        max_documents=args.max_documents,
        page_size=args.page_size,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        max_retries=args.max_retries,
        out_dir=args.out_dir,
        file_format=args.file_format,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run(parse_args())


if __name__ == "__main__":
    main()
