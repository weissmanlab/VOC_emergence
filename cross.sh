work_path=$(cd "$(dirname "$0")"; pwd)
cd "$work_path"
pyfile="plateau_crossing.py"
/usr/bin/env python3 $pyfile --mut 5e-5 --s 0.24 --k 2 --poptype F --runs 100 --lineage