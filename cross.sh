work_path=$(cd "$(dirname "$0")"; pwd)
cd "$work_path"
pyfile="plateau_crossing.py"
/usr/bin/env python3 $pyfile --mut 5e-05 --k 1 --poptype F --s 0.24 --runs 10 --lineage