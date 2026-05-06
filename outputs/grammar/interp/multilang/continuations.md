# Continuations — multilang

Sampling: coda head `T=0.9, top-k=40`; DT head `T=0.9, no top-k` (5 buckets). Mean held-out seed length is used as the target continuation length.

## seed 0 — `zh_birth::zh::Zhou3::001128::33`
original length: 80; generating 37 new tokens after the K=50 prefix

**prefix (first 50 tokens, rendered from real CSV):**
```
  *CHI:	hao3 .
  *MOT:	hei1 .
  *FAT:	wang3 zhong1jian1 qu4 .
  *MOT:	you4 lai2 .
  *MOT:	you4 lai2 .
  *MOT:	kan4 xiao3peng2you3 .
  *MOT:	ai1ya1 kan4 .
  *MOT:	xian1he4 .
  *CHI:	xian1he4 .
  *MOT:	xiao3peng2you3 .
  *MOT:	zhe4 duo1 xiao3peng2you3 .
  *MOT:	ni3 kan4 zhe4 duo1 xiao3peng2you3 qu4 .
  *CHI:	ni3 shui4 zhe4 bian1 a1 .
  *MOT:	ai1ya1 hao3duo1 xiao3peng2you3 .
  *FAT:	xiong2 .
```

**original continuation (ground truth):**
```
  *FAT:	mao1 .
  *MOT:	ni3 kan4 zhe4 shi4 xiong2mao1 .
  *MOT:	wo3 ren4 .
  *MOT:	ni3 kan4 da4xiang4 chu1lai2 .
  *MOT:	wa1 .
  *MOT:	xue3er2 you3 kan4 dao4 da4xiang4 .
  *MOT:	zhe4 shi4 shen2 .
  *MOT:	zhe4 shi4 shen2 a1 .
```

**sample 1:**
```
  *CHI:	tong2 li3mei2 ai1you3 UW fei1bian1ni3jia1bian1 a1 .
  *MOT:	B you3hao3bu4yao4 UW yi2shen2mei2 you3 UW fei1 .
  *MOT:	yao4 ta1hai2 .
  *FAT:	ta1ta1 .
  *CHI:	you3xue3zhe4xie1a1 ni3yao4 en1 .
```

**sample 2:**
```
  *CHI:	o1 yao4 .
  *CHI:	UW .
  *CHI:	a1lai2 UW yi2 UW you3 ming2 .
  *MOT:	you3 .
  *FAT:	zen3xia4lai2jian1you3 .
  *FAT:	you3 UW bing1 xie1ji1 .
  *FAT:	lai2 ming2qi4 .
  *FAT:	ba2 li3 UW fu2yi1qi4 UW fei1 zai4 .
  *CHI:	ta1 UW ge4tong2 .
```

**sample 3:**
```
  *CHI:	you3he1 D UW you3 yi1 .
  *MOT:	ge4li3you2tong2 ming2dai4ta1 han3 UW ta1ta1 liu1 .
  *FAT:	Y bing1 .
  *FAT:	a1 wan2 UW a1zhuan3 re yi4 yang4 zhe4li3li3li3 .
  *CHI:	UW jia1da3 ge4xiao3 .
```

## seed 1 — `zh_dswp::zh::TCCM::010800::13`
original length: 94; generating 37 new tokens after the K=50 prefix

**prefix (first 50 tokens, rendered from real CSV):**
```
  *SHO:	en1 .
  *MOT:	ni3 shuo1hua4 .
  *EXA:	tou2 tong4 .
  *EXA:	ni3 kan4 yi2 xiang4 .
  *EXA:	a1xiu4 .
  *EXA:	ni3 kan4 yi2 xiang4 .
  *TSA:	ni3 na4 duo1 a1 .
  *TSA:	xiao3qiang2 .
  *TSA:	ni3 na2 qu4 na4li3 .
  *TSA:	yao4 na2 qu4 na4li3 .
  *MOT:	yi2 zai4 die2 .
  *MOT:	ni3 kan4 yi2 zai4 die2 fei1ji1 o1 .
  *TSA:	yao4 na2 qu4 na4li3 .
  *ADU:	yi2 jiao1 .
```

**original continuation (ground truth):**
```
  *ADU:	ni3 .
  *ADU:	hao3bu2hao3 .
  *MOT:	lai2 .
  *MOT:	yi2 jiao1 ni3 zuo4 .
  *ADU:	lai2 .
  *ADU:	ni3 gen1 wo3 zuo4 .
  *ADU:	ni3 gen1 wo3 yi1qi3 zuo4 .
  *ADU:	hao3bu2hao3 .
  *MOT:	ni3 qu4 gen1 yi2 bang1mang2 .
  *ADU:	ni3 na2 ji1mu4 gei3 yi2 die2 .
  *ADU:	hao3 .
```

**sample 1:**
```
  *SHO:	li3li3 EHEH AHT UW hua4guo3 .
  *MOT:	yao4zhe4li3 UW .
  *MOT:	BIYAH jiao1 AHBG DEHBBNIY AHAHB .
  *EXA:	D .
  *EXA:	SIH la1 .
  *EXA:	a1 IY .
  *EXA:	S .
  *TSA:	yi1 .
```

**sample 2:**
```
  *SHO:	bu4yao4jiang3da4 wo3ge4yi1 mian4 li3wo3yang4a1ma1 V ge4 UW zai4 mei2 la1shi4zhe4 li3 .
  *SHO:	le1 AHS diao4 .
  *MOT:	DH kuai4 jia1 na3 .
  *MOT:	mei2 .
  *EXA:	you3 zai4 mei2li3 UW li3 .
```

**sample 3:**
```
  *SHO:	li3 K AH .
  *MOT:	ta1ta1 ba4 .
  *MOT:	ge4 .
  *MOT:	AET dou1 AYUW li3 DUW DHEHAO wo3 .
  *EXA:	IHER le1 AHAH YUWD UWT you3 DUW ta1 AH ma1 HH ta1 .
```
