work_path=$(cd "$(dirname "$0")"; pwd)
cd "$work_path"
pyfile="plateau_crossing.py"
/usr/bin/env python3 $pyfile --N 1000000 --mut 1e-6  --rec 0 --s 0.24 --k 1 --poptype C --runs 1 --tstep 1 --plot --lineage 100