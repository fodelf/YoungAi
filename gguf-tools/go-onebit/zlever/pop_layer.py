import struct,os,sys,fcntl
L=int(sys.argv[1])
ld=sys.argv[2] if len(sys.argv)>2 else "/Users/fodelf/ds4-main/gguf/go-onebit/r30/nl86/layers"
man=os.path.join(ld,"zinject_manifest.txt")
lk=open(man+".lock","a")
fcntl.flock(lk,fcntl.LOCK_EX)
try:
    keep=[]
    if os.path.exists(man):
        for ln in open(man):
            p=ln.split()
            if int(p[0])==L:
                osz,n0=int(p[1]),int(p[2])
                if osz>0:
                    f=os.path.join(ld,"dql_L%02d.bin"%L)
                    fh=open(f,"r+b"); fh.truncate(osz); fh.seek(8); fh.write(struct.pack("<I",n0)); fh.close()
                continue
            keep.append(ln)
        open(man,"w").write("".join(keep))
    print("L%d 旧记录摘除(带锁)"%L)
finally:
    fcntl.flock(lk,fcntl.LOCK_UN)
