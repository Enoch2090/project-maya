import re,sys
src=open('/root/autodl-tmp/llama.cpp/ggml/src/ggml-common.h').read()
want={'kmask_iq2xs':('uint8_t',8),'ksigns_iq2xs':('uint8_t',128),'ksigns64':('uint64_t',128),
      'iq2xxs_grid':('uint64_t',256),'iq2s_grid':('uint64_t',1024),'iq3xxs_grid':('uint32_t',256)}
out=[]
for name,(ty,n) in want.items():
    m=re.search(r'GGML_TABLE_BEGIN\(\s*'+ty+r'\s*,\s*'+name+r'\s*,\s*(\d+)\s*\)(.*?)GGML_TABLE_END\(\)',src,re.S)
    if not m: print('MISS',name); sys.exit(1)
    body=m.group(2)
    toks=[t.strip() for t in body.replace('\n',' ').split(',') if t.strip()]
    assert len(toks)==n,(name,len(toks),n)
    vals=[int(t,0) for t in toks]
    out.append((name,ty,n,vals))
    # byte stats
    if name.endswith('_grid'):
        nb=8 if ty=='uint64_t' else 4
        bs=set()
        for v in vals:
            for i in range(nb): bs.add((v>>(8*i))&0xff)
        print(f'# {name}: {n} entries {nb}B, byte min={min(bs)} max={max(bs)} nzero_entries={sum(1 for v in vals for i in range(nb) if ((v>>(8*i))&0xff)==0)}')
hdr=['// auto-generated from ggml-common.h','#pragma once','#include <stdint.h>']
for name,ty,n,vals in out:
    hdr.append(f'static const {ty} H_{name.upper()}[{n}] = {{')
    for i in range(0,n,8):
        hdr.append('  '+', '.join(f'0x{v:0{16 if ty=="uint64_t" else 10 if ty=="uint32_t" else 4}x}' if ty!='uint8_t' else str(v) for v in vals[i:i+8])+',')
    hdr.append('};')
open('tables.h','w').write('\n'.join(hdr)+'\n')
print('wrote tables.h', sum(len(x) for x in hdr))
