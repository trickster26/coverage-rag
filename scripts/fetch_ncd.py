"""Download Medicare National Coverage Determinations from the public CMS Coverage API.

Keeps only the narrative fields. Billing-code tables are left out: they can carry
AMA-licensed CPT content, and the assistant does not need them.
"""

import json
import time
import urllib.request
from pathlib import Path

API = "https://api.coverage.cms.gov/v1"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "cms_ncd.jsonl"
FIELDS = ["document_display_id", "title", "effective_date", "benefit_category",
          "item_service_description", "indications_limitations"]


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{API}{path}", timeout=30) as response:
        return json.load(response)


def main() -> None:
    index = get("/reports/national-coverage-ncd/")["data"]
    with OUT.open("w") as out:
        for row in index:
            doc = get(row["url"])["data"][0]
            out.write(json.dumps({k: doc.get(k, "") for k in FIELDS}) + "\n")
            time.sleep(0.2)
    print(f"wrote {len(index)} NCDs to {OUT}")


if __name__ == "__main__":
    main()
