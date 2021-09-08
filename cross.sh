work_path=$(cd "$(dirname "$0")"; pwd)
cd "$work_path"
pyfile="plateau_crossing.py"
/usr/bin/env python3 $pyfile --N 1000 --mut 1e-3  --rec 0 --s 0.24 --k 1 --poptype C --runs 100 --tmax 100