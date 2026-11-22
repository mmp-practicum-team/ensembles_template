"""Check whether commits on main are associated with merged pull requests."""

import json
import os
import sys
import urllib.error
import urllib.request


BASE = os.environ.get("GITHUB_API_URL", "https://api.github.com")
REPO = os.environ["GITHUB_REPOSITORY"]
TOKEN = os.environ["GH_TOKEN"]


def get_pages(path):
    url = f"{BASE}/repos/{REPO}/{path}"
    while url:
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)
            if not isinstance(data, list):
                raise ValueError("GitHub API вернул не список")
            yield from data
            url = None
            for link in response.headers.get("Link", "").split(","):
                if 'rel="next"' in link:
                    url = link.split(";")[0].strip().strip("<>")
                    break


def main():
    commits = list(get_pages("commits?sha=main&per_page=100"))
    unmatched = []
    for commit in commits:
        sha = commit["sha"]
        prs = get_pages(f"commits/{sha}/pulls?per_page=100")
        if not any(pr.get("merged_at") and pr.get("base", {}).get("ref") == "main" for pr in prs):
            unmatched.append(commit)

    # A repository created from a GitHub template has a single root commit
    # containing the template. It is not a student's direct push.
    unmatched = [commit for commit in unmatched if commit["parents"]]
    print(f"Проверено коммитов в main: {len(commits)}")
    print(f"Коммитов без связанного влитого в main PR: {len(unmatched)}")
    for commit in unmatched:
        message = commit['commit']['message'].splitlines()
        print(f"  {commit['sha'][:12]} {message[0] if message else '(пустое сообщение)'}")
        print(f"    {commit['html_url']}")
    if not unmatched:
        print("OK: коммитов без связанного PR не обнаружено.")

    prs = list(get_pages("pulls?state=closed&base=main&per_page=100"))
    merged = [pr for pr in prs if pr.get("merged_at")]
    empty = [pr for pr in merged if not (pr.get("body") or "").strip()]
    print(f"Проверено влитых в main PR: {len(merged)}")
    print(f"PR с пустым описанием: {len(empty)}")
    for pr in empty:
        print(f"  #{pr['number']} {pr['title']}: {pr['html_url']}")
    if not empty:
        print("OK: пустых описаний влитых PR не обнаружено.")
    if not merged:
        print("Влитых PR пока нет: факт разработки в ветках не подтверждён.")
    return int(bool(unmatched or empty))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (urllib.error.URLError, KeyError, ValueError) as exc:
        print(f"Не удалось проверить историю через GitHub API: {exc}", file=sys.stderr)
        sys.exit(2)
