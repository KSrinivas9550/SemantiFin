"""Small SEC submissions downloader. Respect SEC fair-access policy and identify yourself."""
import argparse, json, time
from pathlib import Path
import requests

def main():
    p=argparse.ArgumentParser(); p.add_argument("--cik",required=True); p.add_argument("--user-agent",required=True,help="Name email@example.com"); p.add_argument("--output",default="data/raw/edgar")
    a=p.parse_args(); cik=str(a.cik).zfill(10); url=f"https://data.sec.gov/submissions/CIK{cik}.json"
    r=requests.get(url,headers={"User-Agent":a.user_agent,"Accept-Encoding":"gzip, deflate"},timeout=30); r.raise_for_status()
    Path(a.output).mkdir(parents=True,exist_ok=True); Path(a.output,f"CIK{cik}.json").write_text(json.dumps(r.json(),indent=2),encoding="utf-8"); time.sleep(0.12)
    print("Saved SEC submission metadata.")
if __name__=="__main__": main()
