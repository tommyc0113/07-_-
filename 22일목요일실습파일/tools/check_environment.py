"""설치된 환경을 확인합니다. 버전 차이는 경고이며 검증 통과를 대신하지 않습니다."""
import sys,importlib,subprocess,shutil,json
r={"python":sys.version.split()[0],"executable":sys.executable};missing=[]
for name in ["numpy","pandas","matplotlib"]:
    try:r[name]=importlib.import_module(name).__version__
    except ImportError:missing.append(name)
r["git"]=subprocess.check_output(["git","--version"],text=True).strip() if shutil.which("git") else "미설치"
r["missing_packages"]=missing
print(json.dumps(r,ensure_ascii=False,indent=2))
print("PPT 검수기준: Python3.13.5/NumPy2.3.5/pandas2.2.3/Matplotlib3.10.8. 다르면 기능검사를 다시 확인하세요.")
raise SystemExit(bool(missing))
