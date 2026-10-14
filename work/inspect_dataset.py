"""Inspect the official archive without extracting or executing its contents."""
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

archive = Path(__file__).parent / "data" / "KuaiRand-Pure.tar.gz"
expected_md5 = "0820331067a3784d9691136f772b35a7"
with archive.open("rb") as source:
    actual_md5 = hashlib.file_digest(source, "md5").hexdigest()
if actual_md5 != expected_md5:
    raise ValueError(f"Archive checksum mismatch: {actual_md5}")

report = {"source": "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz", "md5": actual_md5, "files": {}}
with tarfile.open(archive, "r:gz") as bundle:
    for member in bundle.getmembers():
        if not member.isfile() or not member.name.endswith(".csv"):
            continue
        raw = bundle.extractfile(member)
        with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
            reader = csv.DictReader(text)
            info = {"columns": reader.fieldnames, "rows": 0}
            if "/log_" not in member.name:
                for row in reader:
                    if info["rows"] == 0:
                        info["first_row"] = row
                    info["rows"] += 1
                report["files"][member.name] = info
                continue
            dates = collections.Counter()
            labels = collections.Counter()
            clicks = collections.Counter()
            tabs = collections.Counter()
            users, videos = set(), set()
            bad_times = 0
            min_time, max_time = None, None
            for row in reader:
                if info["rows"] == 0:
                    info["first_row"] = row
                info["rows"] += 1
                dates[row["date"]] += 1
                labels[row["long_view"]] += 1
                clicks[row["is_click"]] += 1
                tabs[row["tab"]] += 1
                users.add(row["user_id"])
                videos.add(row["video_id"])
                try:
                    time = int(row["time_ms"])
                    if time <= 0:
                        bad_times += 1
                    else:
                        min_time = time if min_time is None else min(min_time, time)
                        max_time = time if max_time is None else max(max_time, time)
                except ValueError:
                    bad_times += 1
            info.update(dates=dict(sorted(dates.items())), long_view=dict(labels), is_click=dict(clicks), tab=dict(tabs), users=len(users), videos=len(videos), bad_times=bad_times, min_time_ms=min_time, max_time_ms=max_time)
            report["files"][member.name] = info
output = archive.parent / "inspection.json"
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
