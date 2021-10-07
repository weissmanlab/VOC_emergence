work_path=$(cd "$(dirname "$0")"; pwd)
cd "$work_path"
pyfile="plateau_crossing.py"
/usr/bin/env python3 $pyfile --mut 5e-5 --s 0.24 --k 3 --poptype F --runs 3 --plot --lineage