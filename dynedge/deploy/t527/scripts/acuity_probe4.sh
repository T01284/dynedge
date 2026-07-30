#!/usr/bin/env bash
set -euo pipefail
sed -i 's/\r$//' "$0" 2>/dev/null || true
docker exec npu_v1.8.13_wang bash <<'EOS'
set +e
echo ==== IDE ====
ls /root/Vivante_IDE/VivanteIDE5.8.2/cmdtools/
echo ==== nbinfo self ====
/root/nbinfo -b /etc/npu/sample_viplite/MobileNet_V2/network_binary.nb 2>&1 | head -n 40
# nb may not be in container; copy from host workspace later
echo ==== device strings ====
python3 - <<'PY'
import os,re
root='/usr/local/lib/python3.8/dist-packages/acuitylib'
pat=re.compile(r'VIP[A-Z0-9_]*PID0X[A-F0-9]+')
found=set()
for dp,_,fs in os.walk(root):
  for f in fs:
    if f.endswith(('.py','.json','.yml','.yaml','.txt','.cfg')):
      p=os.path.join(dp,f)
      try:
        t=open(p,'r',errors='ignore').read()
      except Exception:
        continue
      found.update(pat.findall(t))
print('\n'.join(sorted(found)[:80]))
print('count',len(found))
PY
EOS
